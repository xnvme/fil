#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <libfil.h>
#include <fil_io.h>
#include <fil_iter.h>
#include <fil_time.h>
#include <fil_util.h>

#ifdef FIL_CUDA_ENABLED
#include <cuda_runtime.h>
#include <cufile.h>
#include <libxnvme_cuda.h>
#endif
#include <libxal.h>
#include <libxnvme.h>

// Waiting for xal-server to publish its index
#define XAL_ATTACH_TIMEOUT_MS 30000
#define XAL_ATTACH_INTERVAL_MS 100

// Overhead for upcie heap beyond buffer data: covers NVMe queue allocations and internal metadata
#define UPCIE_HEAP_OVERHEAD (128 << 20)
#define UPCIE_HEAP_ALIGN (2 << 20)
#define UPCIE_HEAP_SIZE(buf_total) \
	(((buf_total) + UPCIE_HEAP_OVERHEAD + UPCIE_HEAP_ALIGN - 1) & ~(UPCIE_HEAP_ALIGN - 1))

static int
_xnvme_setup(struct fil_iter *iter, struct fil_dev *device, const char *uri)
{
	const char *backend = iter->opts->backend;
	struct xnvme_opts opts = xnvme_opts_default();
	size_t heap_size =
		UPCIE_HEAP_SIZE(iter->buffer_size * (iter->opts->batch_size / iter->n_devs));
	struct xnvme_dev *dev;
	int err;
#ifdef FIL_CUDA_ENABLED
	CUfileError_t status;
#endif

	opts.homi_id = iter->opts->homi_id;

	if (strcmp(backend, "aisio-cpu") == 0) {
		opts.be = "upcie";
		iter->type = FIL_CPU;
	} else if (strcmp(backend, "aisio-p2p") == 0) {
		opts.be = "upcie-cuda";
		opts.device_heap_size = heap_size;
		iter->type = FIL_P2P;
	} else if (strcmp(backend, "aisio-gpu") == 0) {
		opts.be = "upcie-cuda";
		opts.device_heap_size = heap_size;
		iter->type = FIL_GPU;
	} else if (strcmp(backend, "posix") == 0) {
		iter->type = FIL_FILE;
		return 0;
	} else if (strcmp(backend, "cufile") == 0) {
		iter->type = FIL_FILE;
#ifdef FIL_CUDA_ENABLED
		status = cuFileDriverOpen();
		if (status.err != CU_FILE_SUCCESS) {
			fprintf(stderr, "Could not open cuFile driver: %d\n", status.err);
			return status.err;
		}
#endif
		return 0;
	} else {
		fprintf(stderr, "Invalid backend: %s\n", backend);
		return EINVAL;
	}

	dev = xnvme_dev_open(uri, &opts);
	if (!dev) {
		err = errno;
		fprintf(stderr, "xnvme_dev_open(%s): %d\n", opts.be, err);
		if (err == ENOENT) {
			fprintf(stderr, "Is a homi server running with --homi-id %u?\n",
				opts.homi_id);
		}
		return err;
	}

	err = xnvme_dev_derive_geo(dev);
	if (err) {
		xnvme_dev_close(dev);
		fprintf(stderr, "xnvme_dev_derive_geo(): %d\n", err);
		return err;
	}
	device->nsid = xnvme_dev_get_nsid(dev);

	if (iter->type == FIL_GPU) {
#ifdef FIL_CUDA_ENABLED
		device->cuda_queues =
			malloc(sizeof(*device->cuda_queues) * iter->opts->gpu_nqueues);
		if (!device->cuda_queues) {
			err = errno;
			fprintf(stderr, "malloc(cuda_queues): %d\n", err);
			goto gpu_err_close;
		}

		for (uint32_t q = 0; q < iter->opts->gpu_nqueues; q++) {
			err = xnvme_cuda_queue_create(dev, iter->opts->queue_depth, 0,
						      &device->cuda_queues[q]);
			if (err) {
				fprintf(stderr, "xnvme_cuda_queue_create(): %d\n", err);
				for (uint32_t k = 0; k < q; k++) {
					xnvme_cuda_queue_destroy(dev, device->cuda_queues[k]);
				}
				goto gpu_err_free_queues;
			}
		}

		err = cudaMalloc((void **)&device->cuda_queues_dev,
				 sizeof(*device->cuda_queues) * iter->opts->gpu_nqueues);
		if (err) {
			fprintf(stderr, "cudaMalloc(cuda_queues_dev): %d\n", err);
			goto gpu_err_destroy_queues;
		}

		err = cudaMemcpy(device->cuda_queues_dev, device->cuda_queues,
				 sizeof(*device->cuda_queues) * iter->opts->gpu_nqueues,
				 cudaMemcpyHostToDevice);
		if (err) {
			fprintf(stderr, "cudaMemcpy(cuda_queues_dev): %d\n", err);
			cudaFree(device->cuda_queues_dev);
			goto gpu_err_destroy_queues;
		}
#endif
	} else {
		err = xnvme_queue_init(dev, iter->opts->queue_depth, 0, &device->queue);
		if (err) {
			xnvme_dev_close(dev);
			fprintf(stderr, "xnvme_queue_init(): %d\n", err);
			return err;
		}

		err = xnvme_queue_set_cb(device->queue, fil_io_cb, &device->io_errors);
		if (err) {
			fprintf(stderr, "xnvme_queue_set_cb(): %d\n", err);
			xnvme_queue_term(device->queue);
			device->queue = NULL;
			xnvme_dev_close(dev);
			return err;
		}
	}

	device->dev = dev;
	return 0;

#ifdef FIL_CUDA_ENABLED
gpu_err_destroy_queues:
	for (uint32_t k = 0; k < iter->opts->gpu_nqueues; k++) {
		xnvme_cuda_queue_destroy(dev, device->cuda_queues[k]);
	}
gpu_err_free_queues:
	free(device->cuda_queues);
	device->cuda_queues = NULL;
gpu_err_close:
	xnvme_dev_close(dev);
	return err;
#endif
}

static int
find_buffer_size(struct xal *FIL_UNUSED(xal), struct xal_inode *inode, void *cb_args,
		 int FIL_UNUSED(level))
{
	struct fil_stats *stats = (struct fil_stats *)cb_args;

	if (xal_inode_is_file(inode)) {
		stats->n_files++;
		stats->avg_file_size += inode->size;
		if (inode->size > stats->max_file_size) {
			stats->max_file_size = inode->size;
		}
	}
	return 0;
}

// Waits while xal-server is still publishing the index
static int
_xal_attach(const char *shm_name, struct xal **xal)
{
	struct timespec interval = {.tv_nsec = XAL_ATTACH_INTERVAL_MS * 1000000L};
	int err;

	for (uint32_t waited = 0;; waited += XAL_ATTACH_INTERVAL_MS) {
		err = xal_from_shm(shm_name, xal);
		if ((err != -EAGAIN && err != -ESTALE) || waited >= XAL_ATTACH_TIMEOUT_MS) {
			break;
		}
		nanosleep(&interval, NULL);
	}

	switch (err) {
	case 0:
		return 0;
	case -ENOENT:
		fprintf(stderr, "No xal index published under %s; is xal-server running?\n",
			shm_name);
		break;
	case -EPROTO:
		fprintf(stderr, "The xal index under %s was published by an incompatible xal\n",
			shm_name);
		break;
	default:
		fprintf(stderr, "xal_from_shm(%s): %d\n", shm_name, err);
	}
	return -err;
}

// xal_get_inode() cannot resolve the root of the index, so that is matched by path
static int
_find_data_dir(struct xal *xal, char *data_dir, struct xal_inode **out)
{
	char root_path[XAL_INODE_PATH_MAXLEN + 1];
	struct xal_inode *root = xal_get_root(xal);
	int err;

	err = xal_inode_path(xal, root, root_path, sizeof(root_path));
	if (err < 0) {
		fprintf(stderr, "xal_inode_path(root): %d\n", err);
		return -err;
	}

	if (strcmp(data_dir, root_path) == 0) {
		*out = root;
	} else {
		err = xal_get_inode(xal, data_dir, out);
		if (err) {
			fprintf(stderr, "Couldn't find %s in the index of %s: %d\n", data_dir,
				root_path, err);
			return -err;
		}
	}

	if (!xal_inode_is_dir(*out)) {
		fprintf(stderr, "Not a directory: %s\n", data_dir);
		return ENOTDIR;
	}
	return 0;
}

static int
_xal_setup(struct fil_iter *iter, struct fil_dev *device)
{
	struct fil_stats dev_stats = {0};
	struct xal *xal;
	uint32_t xal_blksize;
	uint64_t buffer_size;
	int err;

	err = _xal_attach(iter->opts->xal_shm, &xal);
	if (err) {
		return err;
	}

	err = _find_data_dir(xal, iter->opts->data_dir, &device->root_inode);
	if (err) {
		xal_close(xal);
		return err;
	}

	// Walk the dataset to populate the summary stats (n_files, max/avg file
	// size) and derive buffer_size from the true largest file.
	err = xal_walk(xal, device->root_inode, find_buffer_size, &dev_stats);
	if (err) {
		fprintf(stderr, "xal_walk(find_buffer_size): %d\n", err);
		xal_close(xal);
		return -err;
	}
	if (!dev_stats.n_files) {
		fprintf(stderr, "No files in %s\n", iter->opts->data_dir);
		xal_close(xal);
		return ENOENT;
	}

	iter->stats->n_files += dev_stats.n_files;
	iter->stats->avg_file_size += dev_stats.avg_file_size;
	if (dev_stats.max_file_size > iter->stats->max_file_size) {
		iter->stats->max_file_size = dev_stats.max_file_size;
	}

	// Align to the file system block size
	xal_blksize = xal_get_sb_blocksize(xal);
	buffer_size = (1 + ((dev_stats.max_file_size - 1) / xal_blksize)) * xal_blksize;
	if (buffer_size > iter->buffer_size) {
		iter->buffer_size = buffer_size;
	}

	// The index keeps each directory's entries sorted by name, so the position
	// of a class directory is a stable label.
	device->xal = xal;
	return 0;
}

static int
_create_entries(struct fil_iter *iter)
{
	struct fil_entry *entries;
	struct xal_inode *inode;
	struct xal *xal = iter->devs[0]->xal;
	struct xal_dentries root_dentries = iter->devs[0]->root_inode->content.dentries;
	uint64_t n_entries = 0;
	uint32_t n_files;
	int err;
	int k;

	for (uint32_t i = 0; i < root_dentries.count; i++) {
		inode = xal_inode_at(xal, root_dentries.inodes_idx + i);
		n_entries += inode->content.dentries.count;
	}

	entries = malloc(sizeof(struct fil_entry) * n_entries);
	if (!entries) {
		err = errno;
		fprintf(stderr, "Could not allocate entries: %d\n", err);
		return err;
	}

	k = 0;
	for (uint32_t i = 0; i < root_dentries.count; i++) {
		inode = xal_inode_at(xal, root_dentries.inodes_idx + i);
		n_files = inode->content.dentries.count;

		for (uint32_t j = 0; j < n_files; j++) {
			entries[k].dir = i;
			entries[k].file = j;
			k++;
		}
	}

	iter->data->entries = entries;
	iter->data->n_entries = n_entries;

	return 0;
}

// cuFile always reads into GPU memory, posix only with copy_to_gpu
static bool
_file_buffers_on_device(struct fil_iter *iter)
{
	return (strcmp(iter->opts->backend, "cufile") == 0) || iter->opts->copy_to_gpu;
}

static int
_alloc(struct fil_iter *iter, uint32_t n_buffers)
{
	int err;
	iter->output = malloc(sizeof(struct fil_output));
	if (!iter->output) {
		err = errno;
		fprintf(stderr, "Could not allocate output struct: %d\n", err);
	}
	iter->output->n_buffers = n_buffers;

	iter->output->buffers = malloc(sizeof(void *) * iter->output->n_buffers);
	if (!iter->output->buffers) {
		err = errno;
		fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
		return err;
	}

	iter->output->labels = malloc(sizeof(uint32_t) * iter->output->n_buffers);
	if (!iter->output->labels) {
		err = errno;
		fprintf(stderr, "Could not allocate array of labels: %d\n", err);
		return err;
	}

	iter->output->buf_len = malloc(sizeof(uint64_t) * iter->output->n_buffers);
	if (!iter->output->buf_len) {
		err = errno;
		fprintf(stderr, "Could not allocate array of buffer lengths: %d\n", err);
		return err;
	}
	memset(iter->output->buf_len, 0, sizeof(uint64_t) * iter->output->n_buffers);

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct fil_dev *device = iter->devs[i];
		device->n_buffers = iter->output->n_buffers / iter->n_devs;
		device->buf = 0;
		device->buffers = malloc(sizeof(void *) * device->n_buffers);
		if (!device->buffers) {
			err = errno;
			fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
			return err;
		}
		if (iter->type == FIL_CPU && iter->opts->copy_to_gpu) {
			device->gpu_buffers = malloc(sizeof(void *) * device->n_buffers);
			if (!device->gpu_buffers) {
				err = errno;
				fprintf(stderr, "Could not allocate array of GPU buffers: %d\n",
					err);
				return err;
			}
		}
		for (uint32_t j = 0; j < device->n_buffers; j++) {
			switch (iter->type) {
			case FIL_GPU:
			case FIL_P2P:
				device->buffers[j] =
					xnvme_buf_alloc(device->dev, iter->buffer_size);
				break;
			case FIL_CPU:
				device->buffers[j] =
					xnvme_buf_alloc(device->dev, iter->buffer_size);
#ifdef FIL_CUDA_ENABLED
				if (device->buffers[j] && iter->opts->copy_to_gpu) {
					err = cudaMalloc(&device->gpu_buffers[j],
							 iter->buffer_size);
					if (err) {
						fprintf(stderr,
							"Could not allocate gpu_buffers[%d]: %d\n",
							i, err);
						return err;
					}
				}
#endif
				break;
			case FIL_FILE:
				if (!_file_buffers_on_device(iter)) {
					// Aligned, as the reads are O_DIRECT unless buffered
					err = posix_memalign(&device->buffers[j],
							     xal_get_sb_blocksize(device->xal),
							     iter->buffer_size);
					if (err) {
						fprintf(stderr,
							"Could not allocate buffers[%d]: %d\n", i,
							err);
						return err;
					}
					break;
				}
#ifdef FIL_CUDA_ENABLED
				err = cudaMalloc(&device->buffers[j], iter->buffer_size);
				if (err) {
					fprintf(stderr, "Could not allocate buffers[%d]: %d\n", i,
						err);
					return err;
				}
				if (strcmp(iter->opts->backend, "cufile") == 0 &&
				    iter->opts->register_bufs) {
					CUfileError_t fstatus = cuFileBufRegister(
						device->buffers[j], iter->buffer_size, 0);
					if (fstatus.err != CU_FILE_SUCCESS) {
						fprintf(stderr,
							"cuFileBufRegister(buffers[%d]): %d\n", i,
							fstatus.err);
						return fstatus.err;
					}
				}
#endif
				break;
			}
			if (!device->buffers[j]) {
				err = errno;
				fprintf(stderr, "Could not allocate buffers[%d]: %d\n", i, err);
				if (iter->type == FIL_CPU) {
					fprintf(stderr,
						"The homi server's host heap must hold %u buffers "
						"of %lu bytes\n",
						iter->opts->batch_size, iter->buffer_size);
				}
				return err;
			}
			if (iter->type == FIL_CPU && iter->opts->copy_to_gpu) {
				iter->output->buffers[j + i * device->n_buffers] =
					device->gpu_buffers[j];
			} else {
				iter->output->buffers[j + i * device->n_buffers] =
					device->buffers[j];
			}
		}

		if (iter->type == FIL_FILE) {
			device->file_io = malloc(sizeof(struct fil_file_io));
			if (!device->file_io) {
				err = errno;
				fprintf(stderr, "Could not allocate IO struct: %d\n", err);
				return err;
			}
			device->file_io->buffer = NULL;
			if (iter->opts->copy_to_gpu) {
				device->file_io->buffer = malloc(iter->buffer_size);
				if (!device->file_io->buffer) {
					err = errno;
					fprintf(stderr, "Could not allocate bounce buffer: %d\n",
						err);
					return err;
				}
			}
		} else if (iter->type == FIL_GPU) {
#ifdef FIL_CUDA_ENABLED
			uint32_t n_cmds =
				(iter->buffer_size / iter->opts->iosize) * device->n_buffers;

			device->gpu_io.n_io = 0;

			err = cudaMallocHost((void **)&device->gpu_io.cmds_host,
					     sizeof(struct xnvme_spec_cmd) * n_cmds);
			if (err) {
				fprintf(stderr, "Could not allocate gpu_io.cmds_host: %d\n", err);
				return err;
			}

			// Pre-fill the fields that are constant for the whole run so the
			// per-batch hot loop only writes slba/nlb/prp1.
			memset(device->gpu_io.cmds_host, 0,
			       sizeof(struct xnvme_spec_cmd) * n_cmds);

			for (uint32_t c = 0; c < n_cmds; c++) {
				device->gpu_io.cmds_host[c].common.opcode =
					XNVME_SPEC_NVM_OPC_READ;
				device->gpu_io.cmds_host[c].common.nsid = device->nsid;
			}

			err = cudaMalloc((void **)&device->gpu_io.cmds_dev,
					 sizeof(struct xnvme_spec_cmd) * n_cmds);
			if (err) {
				cudaFreeHost(device->gpu_io.cmds_host);
				device->gpu_io.cmds_host = NULL;
				fprintf(stderr, "Could not allocate gpu_io.cmds_dev: %d\n", err);
				return err;
			}

			device->gpu_io.prp1_base = malloc(sizeof(uint64_t) * device->n_buffers);
			if (!device->gpu_io.prp1_base) {
				err = errno;
				cudaFree(device->gpu_io.cmds_dev);
				device->gpu_io.cmds_dev = NULL;
				cudaFreeHost(device->gpu_io.cmds_host);
				device->gpu_io.cmds_host = NULL;
				fprintf(stderr, "Could not allocate gpu_io.prp1_base: %d\n", err);
				return err;
			}

			for (uint32_t j = 0; j < device->n_buffers; j++) {
				uint64_t prp1;
				err = xnvme_buf_vtophys(device->dev, device->buffers[j], &prp1);
				if (err) {
					free(device->gpu_io.prp1_base);
					device->gpu_io.prp1_base = NULL;
					cudaFree(device->gpu_io.cmds_dev);
					device->gpu_io.cmds_dev = NULL;
					cudaFreeHost(device->gpu_io.cmds_host);
					device->gpu_io.cmds_host = NULL;
					fprintf(stderr, "xnvme_buf_vtophys(): %d\n", err);
					return err;
				}
				device->gpu_io.prp1_base[j] = prp1;
			}
#endif
		}
	}

#ifdef FIL_CUDA_ENABLED
	if (iter->opts->async) {
		iter->cufile_io = malloc(sizeof(struct fil_cufile_io));
		if (!iter->cufile_io) {
			err = errno;
			fprintf(stderr, "Could not allocate cuFile IO struct: %d\n", err);
			return err;
		}

		iter->cufile_io->descr = malloc(sizeof(CUfileDescr_t) * iter->opts->batch_size);
		if (!iter->cufile_io->descr) {
			err = errno;
			fprintf(stderr, "Could not allocate cuFile descriptors: %d\n", err);
			return err;
		}

		iter->cufile_io->handle = malloc(sizeof(CUfileHandle_t) * iter->opts->batch_size);
		if (!iter->cufile_io->handle) {
			err = errno;
			fprintf(stderr, "Could not allocate cuFile handles: %d\n", err);
			return err;
		}

		iter->cufile_io->expected = malloc(sizeof(size_t) * iter->opts->batch_size);
		if (!iter->cufile_io->expected) {
			err = errno;
			fprintf(stderr, "Could not allocate array of expected values: %d\n", err);
			return err;
		}

		iter->cufile_io->actual = malloc(sizeof(ssize_t) * iter->opts->batch_size);
		if (!iter->cufile_io->actual) {
			err = errno;
			fprintf(stderr, "Could not allocate array of actual values: %d\n", err);
			return err;
		}

		iter->cufile_io->streams = malloc(sizeof(cudaStream_t) * iter->opts->batch_size);
		if (!iter->cufile_io->streams) {
			err = errno;
			fprintf(stderr, "Could not allocate array of CUDA Streams: %d\n", err);
			return err;
		}

		for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
			err = cudaStreamCreateWithFlags(&iter->cufile_io->streams[i],
							cudaStreamNonBlocking);
			if (err) {
				fprintf(stderr, "Could not setup CUDA Stream, err: %d\n", err);
				return err;
			}
		}
	}
#endif

	iter->data = malloc(sizeof(struct fil_data));
	if (!iter->data) {
		err = errno;
		fprintf(stderr, "Could not allocate data: %d\n", err);
		return err;
	}
	memset(iter->data, 0, sizeof(struct fil_data));

	return 0;
}

void
fil_term(struct fil_iter *iter)
{
	FIL_TIME_FREE(iter);
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct fil_dev *device = iter->devs[i];
		switch (iter->type) {
		case FIL_GPU:
#ifdef FIL_CUDA_ENABLED
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				xnvme_buf_free(device->dev, device->buffers[j]);
			}
			cudaFree(device->cuda_queues_dev);
			for (uint32_t q = 0; device->cuda_queues && q < iter->opts->gpu_nqueues;
			     q++) {
				xnvme_cuda_queue_destroy(device->dev, device->cuda_queues[q]);
			}
			free(device->cuda_queues);
			cudaFreeHost(device->gpu_io.cmds_host);
			cudaFree(device->gpu_io.cmds_dev);
			free(device->gpu_io.prp1_base);
#endif
			break;
		case FIL_P2P:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				xnvme_buf_free(device->dev, device->buffers[j]);
			}
			xnvme_queue_term(device->queue);
			break;
		case FIL_CPU:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				xnvme_buf_free(device->dev, device->buffers[j]);
#ifdef FIL_CUDA_ENABLED
				if (device->gpu_buffers) {
					cudaFree(device->gpu_buffers[j]);
				}
#endif
			}
			free(device->gpu_buffers);
			xnvme_queue_term(device->queue);
			break;
		case FIL_FILE:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				if (!_file_buffers_on_device(iter)) {
					free(device->buffers[j]);
					continue;
				}
#ifdef FIL_CUDA_ENABLED
				if (strcmp(iter->opts->backend, "cufile") == 0 &&
				    iter->opts->register_bufs) {
					cuFileBufDeregister(device->buffers[j]);
				}
				cudaFree(device->buffers[j]);
#endif
			}
			break;
		}
		xal_close(device->xal);
		xnvme_dev_close(device->dev);
#ifdef FIL_CUDA_ENABLED
		cuFileDriverClose();
#endif
		if (device->file_io) {
			free(device->file_io->buffer);
			free(device->file_io);
		}
		free(device->buffers);
		free(device);
	}
#ifdef FIL_CUDA_ENABLED
	if (iter->cufile_io) {
		free(iter->cufile_io->descr);
		free(iter->cufile_io->handle);
		free(iter->cufile_io->expected);
		free(iter->cufile_io->actual);
		for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
			cudaStreamDestroy(iter->cufile_io->streams[i]);
		}
		free(iter->cufile_io->streams);
	}
#endif
	if (iter->data) {
		free(iter->data->entries);
		free(iter->data);
	}
	free(iter->opts);
	if (iter->output) {
		free(iter->output->buffers);
		free(iter->output->labels);
		free(iter->output->buf_len);
		free(iter->output);
	}
	free(iter->stats);
	free(iter);
}

static int
_init_stats(struct fil_stats **stats)
{
	int err;
	struct fil_stats *_stats = malloc(sizeof(struct fil_stats));
	if (!_stats) {
		err = errno;
		fprintf(stderr, "Could not allocate stats: %d\n", err);
		return err;
	}
	memset(_stats, 0, sizeof(struct fil_stats));
	*stats = _stats;
	return 0;
}

int
fil_init(struct fil_iter **iter, char **dev_uris, uint32_t n_devs, struct fil_opts *opts)
{
	struct fil_iter *_iter;
	srand(time(NULL));
	int err;

	if (n_devs != 1) {
		fprintf(stderr, "fil reads from one device, not %u\n", n_devs);
		return EINVAL;
	}

	if (opts->data_dir[0] != '/') {
		fprintf(stderr, "data_dir must be an absolute path\n");
		return EINVAL;
	}

	if (opts->xal_shm[0] == '\0') {
		fprintf(stderr, "xal_shm is required\n");
		return EINVAL;
	}

	if (opts->batch_size % n_devs != 0) {
		fprintf(stderr, "Batch size (%u) not divisible by number of devices (%u)\n",
			opts->batch_size, n_devs);
	}

#ifndef FIL_CUDA_ENABLED
	if (strcmp(opts->backend, "aisio-gpu") == 0 || strcmp(opts->backend, "cufile") == 0 ||
	    opts->copy_to_gpu) {
		fprintf(stderr, "fil is built without CUDA, which %s needs\n",
			opts->copy_to_gpu ? "copy_to_gpu" : opts->backend);
		return ENOTSUP;
	}
#endif

	if (!opts->homi_id && strncmp(opts->backend, "aisio-", strlen("aisio-")) == 0) {
		fprintf(stderr, "homi_id is required for the aisio backends\n");
		return EINVAL;
	}

	if (opts->buffered && strcmp(opts->backend, "posix") != 0) {
		fprintf(stderr, "opts->buffered == true is only compatible with POSIX backend");
		return EINVAL;
	}

	if (opts->async && strcmp(opts->backend, "cufile") != 0) {
		fprintf(stderr, "opts->async == true is only compatible with cuFile backend");
		return EINVAL;
	}

	if (opts->register_bufs && strcmp(opts->backend, "cufile") != 0) {
		fprintf(stderr,
			"opts->register_bufs == true is only compatible with cuFile backend");
		return EINVAL;
	}

	if (opts->copy_to_gpu && strcmp(opts->backend, "aisio-cpu") != 0 &&
	    strcmp(opts->backend, "posix") != 0) {
		fprintf(stderr, "opts->copy_to_gpu == true is only compatible with the aisio-cpu "
				"and posix "
				"backends\n");
		return EINVAL;
	}

	_iter = malloc(sizeof(struct fil_iter));
	if (!_iter) {
		err = errno;
		fprintf(stderr, "Could not allocate iter: %d\n", err);
		return err;
	}
	memset(_iter, 0, sizeof(struct fil_iter));

	_iter->opts = malloc(sizeof(struct fil_opts));
	if (!_iter->opts) {
		err = errno;
		fprintf(stderr, "Conld not allocate opts: %d\n", err);
		return err;
	}
	memcpy(_iter->opts, opts, sizeof(*opts));

	_iter->devs = malloc(sizeof(struct fil_dev *) * n_devs);
	if (!_iter->devs) {
		err = errno;
		fprintf(stderr, "Could not allocate devices: %d\n", err);
		fil_term(_iter);
		return err;
	}

	err = _init_stats(&_iter->stats);
	if (err) {
		fil_term(_iter);
		return err;
	}

	// The largest file sizes the buffers, and with them the heap the device is opened with
	for (uint32_t i = 0; i < n_devs; i++) {
		struct fil_dev *device = malloc(sizeof(struct fil_dev));
		if (!device) {
			err = errno;
			fprintf(stderr, "Could not allocate handle for %s: %d\n", dev_uris[i],
				err);
			fil_term(_iter);
			return err;
		}
		memset(device, 0, sizeof(struct fil_dev));

		err = _xal_setup(_iter, device);
		if (err) {
			fprintf(stderr, "XAL setup failed for %s: %d\n", dev_uris[i], err);
			free(device);
			fil_term(_iter);
			return err;
		}
		_iter->devs[i] = device;
		_iter->n_devs++;
	}
	_iter->stats->avg_file_size /= _iter->stats->n_files;

	for (uint32_t i = 0; i < n_devs; i++) {
		err = _xnvme_setup(_iter, _iter->devs[i], dev_uris[i]);
		if (err) {
			fprintf(stderr, "xNVMe setup failed for %s: %d\n", dev_uris[i], err);
			fil_term(_iter);
			return err;
		}
	}

	err = _alloc(_iter, _iter->opts->batch_size);
	if (err) {
		fil_term(_iter);
		return err;
	}
	switch (_iter->type) {
	case FIL_GPU:
		_iter->io_fn = fil_gpu_submit;
		err = FIL_TIME_ALLOC(_iter);
		if (err) {
			fil_term(_iter);
			return err;
		}
		break;
	case FIL_P2P:
	case FIL_CPU:
		_iter->io_fn = fil_cpu_submit;
		break;
	case FIL_FILE:
		if (_iter->opts->async) {
			_iter->io_fn = fil_cufile_async_submit;
		} else {
			_iter->io_fn = fil_file_submit;
		}
		break;
	}

	// Create an entry for every file in every directory
	err = _create_entries(_iter);
	if (err) {
		fil_term(_iter);
		return err;
	}

	FIL_SHUFFLE(_iter->data->entries, struct fil_entry, _iter->data->n_entries, uint64_t);

	(*iter) = _iter;

	return 0;
}

int
fil_next(struct fil_iter *iter, struct fil_output **output)
{
	int err;

	if (iter->data->entries && iter->data->index >= iter->data->n_entries) {
		iter->data->index = 0;
		FIL_SHUFFLE(iter->data->entries, struct fil_entry, iter->data->n_entries,
			    uint64_t);
	}

	err = iter->io_fn(iter);
	if (err) {
		fprintf(stderr, "Data reading failed, err: %d\n", err);
		return err;
	}

	*output = iter->output;

	return 0;
}

struct fil_opts
fil_opts_default()
{
	struct fil_opts opts = {.data_dir = "",
				.xal_shm = "",
				.backend = "aisio-cpu",
				.homi_id = 0,
				.iosize = 4096,
				.gpu_nqueues = 128,
				.queue_depth = 1024,
				.batch_size = 1,
				.buffered = false,
				.async = false,
				.register_bufs = false,
				.copy_to_gpu = false};

	return opts;
}

struct fil_stats *
fil_get_stats(struct fil_iter *iter)
{
	return iter->stats;
}
