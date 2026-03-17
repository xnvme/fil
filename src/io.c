#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <libsil.h>
#include <sil_io.h>
#include <sil_iter.h>

#include <libxal.h>
#include <libxnvme.h>

#include <cuda_runtime.h>
#include <cufile.h>

#define GPU_WARPSIZE 32

#define ELAPSED(s, e) \
	((double)((e).tv_sec - (s).tv_sec) + (double)((e).tv_nsec - (s).tv_nsec) / 1e9)

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
	if (work->ranges[work->cur_range].slba >= work->ranges[work->cur_range].elba) {
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

int
sil_cpu_submit(struct sil_iter *iter)
{
	struct sil_entry entry;
	struct xal_inode dir;
	struct xal_inode file;
	struct xal_extent extent, next_extent;
	uint64_t nblocks, nbytes, blocksize, next_slba;
	uint32_t xal_blksize;
	struct timespec start, end;
	int err;

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		struct sil_dev *device = iter->devs[i];
		blocksize = xnvme_dev_get_geo(device->dev)->lba_nbytes;
		xal_blksize = xal_get_sb_blocksize(device->xal);

		for (uint32_t j = 0; j < device->n_buffers; j++) {
			entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];

			dir = device->root_inode->content.dentries.inodes[entry.dir];
			file = dir.content.dentries.inodes[entry.file];
			extent = file.content.extents.extent[0];
			device->cpu_io->slbas[j] =
			    xal_fsbno_offset(device->xal, extent.start_block) / blocksize;

			nbytes = extent.nblocks * xal_blksize;
			nblocks = nbytes / blocksize;
			iter->output->buf_len[j + i * device->n_buffers] = file.size;
			iter->output->labels[j + i * device->n_buffers] = entry.dir;

			for (uint32_t k = 1; k < file.content.extents.count; k++) {
				next_extent = file.content.extents.extent[k];
				next_slba = xal_fsbno_offset(device->xal, next_extent.start_block) /
					    blocksize;
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
			iter->stats->io += nblocks / (iter->opts->nlb + 1);
			iter->stats->bytes += file.size;
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->prep_time += ELAPSED(start, end);

		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		err = _io_range_submit(device->queue, XNVME_SPEC_NVM_OPC_READ,
					    device->cpu_io->slbas, device->cpu_io->elbas,
					    iter->opts->nlb, iter->opts->nbytes, device->buffers,
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
sil_gpu_submit(struct sil_iter *iter)
{
	struct sil_entry entry;
	struct xal_inode dir;
	struct xal_inode file;
	struct xal_extent extent;
	uint64_t nblocks, nbytes, blocksize, slba, offset, n_io = 0;
	uint32_t dev_id, buf_id, xal_blksize;
	void *buffer;
	struct timespec start, end;
	int err;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		dev_id = i % iter->n_devs;
		struct sil_dev *device = iter->devs[dev_id];
		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];
		blocksize = xnvme_dev_get_geo(device->dev)->lba_nbytes;
		xal_blksize = xal_get_sb_blocksize(device->xal);
		offset = 0;

		entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];
		dir = device->root_inode->content.dentries.inodes[entry.dir];
		file = dir.content.dentries.inodes[entry.file];
		iter->output->buf_len[buf_id + dev_id * device->n_buffers] = file.size;
		iter->output->labels[buf_id + dev_id * device->n_buffers] = entry.dir;
		iter->stats->bytes += file.size;
		for (uint32_t j = 0; j < file.content.extents.count; j++) {
			extent = file.content.extents.extent[j];
			slba = xal_fsbno_offset(device->xal, extent.start_block) / blocksize;
			nbytes = extent.nblocks * xal_blksize;
			nblocks = nbytes / blocksize;
			for (uint64_t k = 0; k < nblocks / (iter->opts->nlb + 1); k++) {
				iter->gpu_io->offsets[n_io] = offset * (iter->opts->nlb + 1);
				iter->gpu_io->slbas[n_io] = slba + k * (iter->opts->nlb + 1);
				iter->gpu_io->buffers[n_io] = buffer;
				iter->gpu_io->devs[n_io] = device->dev;
				n_io++;
				offset++;
			}
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->prep_time += ELAPSED(start, end);
	iter->stats->io += n_io;
	iter->gpu_io->n_io = n_io;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	err = xnvme_gpu_io_submit((n_io + iter->opts->gpu_tbsize - 1) / iter->opts->gpu_tbsize,
				  iter->opts->gpu_tbsize, XNVME_SPEC_NVM_OPC_READ, iter->opts->nlb,
				  iter->opts->nbytes, iter->gpu_io);
	if (err) {
		fprintf(stderr, "Could not launch kernel: %d\n", err);
		return err;
	}

	err = xnvme_gpu_sync();
	if (err) {
		fprintf(stderr, "Error synchronizing kernels: %d\n", err);
		return err;
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += ELAPSED(start, end);
	return 0;
}

int
sil_file_submit(struct sil_iter *iter)
{
	struct sil_entry entry;
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
		struct sil_dev *device = iter->devs[dev_id];
		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];
		prefix = device->file_io->prefix;
		path = device->file_io->path;
		bounce = device->file_io->buffer;
		xal_blksize = xal_get_sb_blocksize(device->xal);

		entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];
		dir = device->root_inode->content.dentries.inodes[entry.dir];
		file = dir.content.dentries.inodes[entry.file];

		iter->output->buf_len[buf_id + dev_id * device->n_buffers] = file.size;
		iter->output->labels[buf_id + dev_id * device->n_buffers] = entry.dir;
		iter->stats->bytes += file.size;
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
sil_gds_async_submit(struct sil_iter *iter)
{
	struct sil_entry entry;
	struct xal_inode dir;
	struct xal_inode file;
	struct timespec start, end;
	struct sil_gds_io *gds_io;
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
		struct sil_dev *device = iter->devs[dev_id];
		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];
		prefix = device->file_io->prefix;
		path = device->file_io->path;

		entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];
		dir = device->root_inode->content.dentries.inodes[entry.dir];
		file = dir.content.dentries.inodes[entry.file];

		iter->output->buf_len[buf_id + dev_id * device->n_buffers] = file.size;
		iter->output->labels[buf_id + dev_id * device->n_buffers] = entry.dir;
		iter->stats->bytes += file.size;
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
