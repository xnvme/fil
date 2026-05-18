#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <libfil.h>
#include <fil_io.h>
#include <fil_iter.h>
#include <fil_time.h>
#include <fil_util.h>

#include <libxal.h>
#include <libxnvme.h>

#include <cuda_runtime.h>
#include <cufile.h>

struct _range {
	uint64_t slba;
	uint64_t elba;
	void *dbuf;
};

struct _work {
	uint32_t opc;
	uint32_t nlb;
	uint64_t nbytes;

	uint32_t n_ranges;
	uint32_t cur_range;

	struct _range *ranges;
	uint32_t errors;
};

static int
_submit(struct _work *work, struct xnvme_cmd_ctx *ctx)
{
	int err;

	ctx->cmd.common.opcode = work->opc;
	ctx->cmd.common.nsid = xnvme_dev_get_nsid(ctx->dev);
	if (work->ranges[work->cur_range].slba > work->ranges[work->cur_range].elba) {
		work->cur_range += 1;
	}
	if (work->cur_range >= work->n_ranges) {
		xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);
		return 0;
	}

	ctx->cmd.nvm.slba = work->ranges[work->cur_range].slba;
	ctx->cmd.nvm.nlb = work->nlb;

retry:
	err = xnvme_cmd_pass(ctx, work->ranges[work->cur_range].dbuf, work->nbytes, NULL, 0);
	if (err == -EBUSY || err == -EAGAIN) {
		xnvme_queue_poke(ctx->async.queue, 0);
		goto retry;
	}
	if (err) {
		printf("Failed to submit, err: %d\n", err);
		xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);
		return err;
	}
	work->ranges[work->cur_range].slba += (work->nlb + 1);
	work->ranges[work->cur_range].dbuf =
	    (uint8_t *)work->ranges[work->cur_range].dbuf + work->nbytes;
	return 0;
}

static void
_cb_fn(struct xnvme_cmd_ctx *ctx, void *cb_arg)
{
	struct _work *work = cb_arg;
	if (xnvme_cmd_ctx_cpl_status(ctx)) {
		xnvme_cmd_ctx_pr(ctx, XNVME_PR_DEF);
		work->errors++;
		xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);
		return;
	}

	if (work->cur_range >= work->n_ranges) {
		xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);
		return;
	}
	_submit(work, ctx);
}

static int
_io_range_submit(struct xnvme_queue *queue, uint32_t opc, uint64_t *slbas, uint64_t *elbas,
		 uint32_t nlb, uint64_t nbytes, void **dbufs, uint32_t n_ranges)
{
	struct _work work = {0};
	struct _range ranges[n_ranges];
	struct _range *range;
	uint32_t n_blocks;
	int err, capacity;

	capacity = xnvme_queue_get_capacity(queue);
	work.nlb = nlb;
	work.n_ranges = n_ranges;
	work.nbytes = nbytes;
	work.opc = opc;
	work.ranges = ranges;

	err = xnvme_queue_set_cb(queue, _cb_fn, &work);
	if (err) {
		printf("Failed to set queue callback, err: %d\n", err);
		return err;
	}

	for (uint32_t i = 0; i < n_ranges; i++) {
		range = &work.ranges[i];
		range->slba = slbas[i];
		range->elba = elbas[i];
		range->dbuf = dbufs[i];
		n_blocks = (range->elba - range->slba) + 1;
		if (n_blocks % (nlb + 1) != 0) {
			printf("n_blocks (%u) is not divisible by nlb + 1 (%u)\n", n_blocks,
				    nlb + 1);
			return -EINVAL;
		}
	}

	for (int i = 0; i < capacity - 1; i++) {
		struct xnvme_cmd_ctx *ctx = xnvme_queue_get_cmd_ctx(queue);
		err = _submit(&work, ctx);
		if (err) {
			break;
		}
	}
	xnvme_queue_drain(queue);
	if (work.errors) {
		return -EIO;
	}
	return err;
}

/**
 * Pull the next file from the shuffled dataset, resolve its directory and inode,
 * and record its length, label and byte count into the output and stats. Returns
 * the file inode and writes the parent directory inode to *dir.
 */
static struct xal_inode
fil_next_file(struct fil_iter *iter, struct fil_dev *device, uint32_t dev_id, uint32_t buf_id,
	      struct xal_inode *dir)
{
	struct fil_entry entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];
	uint32_t slot = buf_id + dev_id * device->n_buffers;
	struct xal_inode file;

	*dir = device->root_inode->content.dentries.inodes[entry.dir];
	file = dir->content.dentries.inodes[entry.file];

	iter->output->buf_len[slot] = file.size;
	iter->output->labels[slot] = entry.dir;
	iter->stats->bytes += file.size;

	return file;
}

/** Convert an extent's starting FS block to a device LBA. */
static uint64_t
fil_extent_slba(struct fil_dev *device, const struct xal_extent *extent, uint64_t blocksize)
{
	return xal_fsbno_offset(device->xal, extent->start_block) / blocksize;
}

int
fil_cpu_submit(struct fil_iter *iter)
{
	struct xal_inode dir;
	struct xal_inode file;
	struct xal_extent extent, next_extent;
	uint64_t nblocks, nbytes, blocksize, next_slba;
	uint32_t xal_blksize, nlb;
	struct timespec start, end;
	int err;

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		struct fil_dev *device = iter->devs[i];
		blocksize = xnvme_dev_get_geo(device->dev)->lba_nbytes;
		xal_blksize = xal_get_sb_blocksize(device->xal);
		nlb = iter->opts->iosize / blocksize - 1;

		for (uint32_t j = 0; j < device->n_buffers; j++) {
			file = fil_next_file(iter, device, i, j, &dir);
			extent = file.content.extents.extent[0];
			device->cpu_io->slbas[j] = fil_extent_slba(device, &extent, blocksize);

			nbytes = extent.nblocks * xal_blksize;
			nblocks = nbytes / blocksize;

			for (uint32_t k = 1; k < file.content.extents.count; k++) {
				next_extent = file.content.extents.extent[k];
				next_slba = fil_extent_slba(device, &next_extent, blocksize);
				if (next_slba != device->cpu_io->slbas[j] + nblocks) {
					fprintf(
					    stderr,
					    "File: %s, in dir: %s, has non contiguous extents\n",
					    file.name, dir.name);
					fprintf(stderr,
						"extent[%d].elba: %lu, extent[%d].slba: %lu\n",
						k - 1, device->cpu_io->slbas[j] + nblocks - 1, k,
						next_slba);
					return ENOTSUP;
				}
				nbytes += next_extent.nblocks * xal_blksize;
				nblocks = nbytes / blocksize;
			}
			device->cpu_io->elbas[j] = device->cpu_io->slbas[j] + nblocks - 1;
			iter->stats->io += nblocks / (nlb + 1);
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->prep_time += ELAPSED(start, end);

		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		err = _io_range_submit(device->queue, XNVME_SPEC_NVM_OPC_READ,
					    device->cpu_io->slbas, device->cpu_io->elbas,
					    nlb, iter->opts->iosize, device->buffers,
					    device->n_buffers);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->io_time += ELAPSED(start, end);
		if (err) {
			fprintf(stderr, "IO failed: %d\n", err);
			return err;
		}
	}
	return 0;
}

int
fil_gpu_submit(struct fil_iter *iter)
{
	struct xal_inode dir, file;
	struct xal_extent extent;
	struct timespec start, end;
	struct fil_dev *device;
	struct xnvme_spec_cmd *cmd;
	uint64_t nblocks, nbytes, slba, offset;
	uint32_t dev_id, buf_id;
	int err;

	uint64_t max_ios_per_buf = iter->buffer_size / iter->opts->iosize;
	uint64_t blocksizes[iter->n_devs];
	uint32_t nlbs[iter->n_devs];
	uint32_t xal_blksizes[iter->n_devs];

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		blocksizes[i] = xnvme_dev_get_geo(iter->devs[i]->dev)->lba_nbytes;
		if (iter->opts->iosize < blocksizes[i] ||
		    iter->opts->iosize % blocksizes[i]) {
			fprintf(stderr,
				"iosize (%lu) must be a multiple of the device LBA size (%lu)\n",
				iter->opts->iosize, blocksizes[i]);
			return EINVAL;
		}
		nlbs[i] = iter->opts->iosize / blocksizes[i] - 1;
		xal_blksizes[i] = xal_get_sb_blocksize(iter->devs[i]->xal);
		iter->devs[i]->gpu_io.n_io = 0;
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	FIL_TIME_BEGIN(iter);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		dev_id = i % iter->n_devs;
		device = iter->devs[dev_id];
		buf_id = device->buf++ % device->n_buffers;
		offset = 0;

		file = fil_next_file(iter, device, dev_id, buf_id, &dir);
		FIL_TIME_TICK(iter, prep_meta_time);

		for (uint32_t j = 0; j < file.content.extents.count; j++) {
			extent = file.content.extents.extent[j];
			slba = fil_extent_slba(device, &extent, blocksizes[dev_id]);
			nbytes = extent.nblocks * xal_blksizes[dev_id];
			nblocks = nbytes / blocksizes[dev_id];

			for (uint64_t k = 0; k < nblocks / (nlbs[dev_id] + 1); k++) {
				cmd = &device->gpu_io.cmds_host[device->gpu_io.n_io];

				if (offset >= max_ios_per_buf) {
					fprintf(stderr,
						"File %s exceeds --max-file-size; "
						"increase it to fit the largest file\n",
						file.name);
					return EINVAL;
				}

				cmd->nvm.slba = slba + k * (nlbs[dev_id] + 1);
				cmd->nvm.nlb = nlbs[dev_id];
				cmd->common.dptr.prp.prp1 =
				    device->gpu_io.prp1_base[buf_id] + offset * iter->opts->iosize;
				device->gpu_io.n_io++;
				offset++;
			}
		}
		FIL_TIME_TICK(iter, prep_cmds_time);
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->prep_time += ELAPSED(start, end);

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		iter->stats->io += iter->devs[i]->gpu_io.n_io;
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	FIL_TIME_BEGIN(iter);
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		device = iter->devs[i];

		if (!device->gpu_io.n_io) {
			continue;
		}
		err = cudaMemcpy(device->gpu_io.cmds_dev, device->gpu_io.cmds_host,
				 sizeof(struct xnvme_spec_cmd) * device->gpu_io.n_io,
				 cudaMemcpyHostToDevice);
		if (err) {
			fprintf(stderr, "cudaMemcpy failed: %d\n", err);
			return err;
		}
	}
	FIL_TIME_TICK(iter, io_memcpy_time);
	FIL_TIME_KERNEL_START(iter);
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct fil_dev *device = iter->devs[i];

		if (!device->gpu_io.n_io) {
			continue;
		}
		gpu_io_launch(device->cuda_queues_dev, iter->opts->gpu_nqueues,
			      device->gpu_io.cmds_dev, device->gpu_io.n_io,
			      iter->opts->queue_depth);
	}
	FIL_TIME_KERNEL_END(iter);
	err = cudaDeviceSynchronize();
	if (err) {
		fprintf(stderr, "cudaDeviceSynchronize failed: %d\n", err);
		return err;
	}
	FIL_TIME_TICK(iter, io_sync_time);
	FIL_TIME_KERNEL_ELAPSED(iter);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += ELAPSED(start, end);
	return 0;
}

int
fil_file_submit(struct fil_iter *iter)
{
	struct xal_inode dir;
	struct xal_inode file;
	struct timespec start, end;
	CUfileError_t status;
	CUfileDescr_t descr;
	CUfileHandle_t fh;
	uint32_t buf_id, dev_id, xal_blksize;
	uint64_t nbytes;
	void *buffer, *bounce;
	char *prefix, *path;
	int fd, flags;
	ssize_t err, bytes_read;
	bool is_gds = strcmp(iter->opts->backend, "gds") == 0;
	flags = O_RDONLY;
	if (is_gds || !iter->opts->buffered) {
		flags |= O_DIRECT;
	}

	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		dev_id = i % iter->n_devs;
		struct fil_dev *device = iter->devs[dev_id];
		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];
		prefix = device->file_io->prefix;
		path = device->file_io->path;
		bounce = device->file_io->buffer;
		xal_blksize = xal_get_sb_blocksize(device->xal);

		file = fil_next_file(iter, device, dev_id, buf_id, &dir);
		iter->stats->io++;

		memcpy(path, prefix, strlen(prefix) + 1);
		strcat(path, "/");
		strcat(path, dir.name);
		strcat(path, "/");
		strcat(path, file.name);

		nbytes = file.size;
		if (!is_gds && !iter->opts->buffered) {
			// POSIX O_DIRECT requires aligned nbytes
			nbytes = (1 + ((file.size - 1) / xal_blksize)) * xal_blksize;
		}

		fd = open(path, flags);
		if (fd == -1) {
			err = errno;
			fprintf(stderr, "Could not open %s, err: %ld\n", path, err);
			return err;
		}

		if (is_gds) {
			memset(&descr, 0, sizeof(CUfileDescr_t));
			descr.handle.fd = fd;
			descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
			status = cuFileHandleRegister(&fh, &descr);
			if (status.err != CU_FILE_SUCCESS) {
				fprintf(stderr, "Could not register file, err: %d\n", status.err);
				close(fd);
				return status.err;
			}
		}

		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->prep_time += ELAPSED(start, end);

		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		bytes_read = 0;
		if (is_gds) {

			bytes_read = cuFileRead(fh, buffer, nbytes, 0, 0);
			if (bytes_read < 0) {
				err = bytes_read;
				fprintf(stderr, "Could not read %s, err: %ld\n", path, err);
				return err;
			}
			if ((uint64_t)bytes_read != nbytes) {
				fprintf(
				    stderr,
				    "Could not read entire file %s, expected: %lu, actual: %ld\n",
				    path, file.size, bytes_read);
				return EIO;
			}
			cuFileHandleDeregister(fh);
		} else {
			do {
				err = read(fd, bounce, nbytes - bytes_read);
				if (err == -1) {
					err = errno;
					fprintf(stderr, "Could not read %s, err: %ld\n", path, err);
					return err;
				}
				if (err == 0) {
					fprintf(stderr,
						"Could not read entire file %s, expected: %lu, "
						"actual: %ld\n",
						path, file.size, bytes_read);
					return EIO;
				}
				bytes_read += err;
			} while ((uint64_t)bytes_read != file.size);

			err = cudaMemcpy(buffer, bounce, file.size, cudaMemcpyHostToDevice);
			if (err) {
				fprintf(stderr, "Could not copy data to GPU memory, err: %ld\n",
					err);
				return err;
			}
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->io_time += ELAPSED(start, end);
		close(fd);
	}
	return 0;
}

int
fil_gds_async_submit(struct fil_iter *iter)
{
	struct xal_inode dir;
	struct xal_inode file;
	struct timespec start, end;
	struct fil_gds_io *gds_io;
	CUfileError_t status;
	uint32_t buf_id, dev_id;
	void *buffer;
	char *prefix, *path;
	int fd, flags, err = 0;
	off_t offset = 0;

	gds_io = iter->gds_io;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		dev_id = i % iter->n_devs;
		struct fil_dev *device = iter->devs[dev_id];
		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];
		prefix = device->file_io->prefix;
		path = device->file_io->path;

		file = fil_next_file(iter, device, dev_id, buf_id, &dir);
		iter->stats->io++;

		memcpy(path, prefix, strlen(prefix) + 1);
		strcat(path, "/");
		strcat(path, dir.name);
		strcat(path, "/");
		strcat(path, file.name);

		flags = O_RDONLY | O_DIRECT;

		fd = open(path, flags);
		if (fd == -1) {
			err = errno;
			fprintf(stderr, "Could not open %s, err: %d\n", path, err);
			return err;
		}

		memset(&gds_io->descr[i], 0, sizeof(CUfileDescr_t));
		gds_io->descr[i].handle.fd = fd;
		gds_io->descr[i].type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
		status = cuFileHandleRegister(&gds_io->handle[i], &gds_io->descr[i]);
		if (status.err != CU_FILE_SUCCESS) {
			fprintf(stderr, "Could not register file, err: %d\n", status.err);
			close(fd);
			return status.err;
		}
		gds_io->expected[i] = file.size;
		gds_io->actual[i] = 0;

		status = cuFileReadAsync(gds_io->handle[i], buffer, &gds_io->expected[i], &offset, &offset, &gds_io->actual[i],
				    gds_io->streams[i]);
		if (status.err != CU_FILE_SUCCESS) {
			fprintf(stderr, "cuFileReadAsync failed, err: %d\n", status.err);
			err = status.err;
			goto teardown;
		}
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->prep_time += ELAPSED(start, end);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		err = cudaStreamSynchronize(gds_io->streams[i]);
		if (err) {
			fprintf(stderr, "Could not synchronize CUDA Stream, err: %d\n", err);
			goto teardown;
		}
		if (gds_io->actual[i] < 0) {
			err = gds_io->actual[i];
			fprintf(stderr, "Reading failed, err: %d\n", err);
			goto teardown;
		}
		if ((size_t)gds_io->actual[i] != gds_io->expected[i]) {
			fprintf(stderr,
				"Could not read entire file, expected: %lu, actual: %lu\n",
				gds_io->expected[i], gds_io->actual[i]);
			err = EIO;
			goto teardown;
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += ELAPSED(start, end);

teardown:
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		fd = gds_io->descr[i].handle.fd;
		cuFileHandleDeregister(gds_io->handle[i]);
		close(fd);
	}
	return err;
}
