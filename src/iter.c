#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

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

// Waiting for a re-index, which walks the whole file system
#define XAL_REINDEX_TIMEOUT_MS 300000
#define XAL_REINDEX_NOTICE_MS 1000

// Reads of the dataset or a batch before giving up on an index that keeps changing
#define FIL_READ_ATTEMPTS 3

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
	opts.nsid = iter->opts->nsid;

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

// Waits until the index is neither being rewritten nor waiting for a re-index
static int
_xal_settle(struct fil_dev *device)
{
	struct timespec interval = {.tv_nsec = XAL_ATTACH_INTERVAL_MS * 1000000L};

	for (uint32_t waited = 0;; waited += XAL_ATTACH_INTERVAL_MS) {
		int seq = xal_get_seq_lock(device->xal);

		if (!(seq & 1) && !xal_is_dirty(device->xal)) {
			device->xal_seq = seq;
			return 0;
		}
		if (waited == XAL_REINDEX_NOTICE_MS) {
			fprintf(stderr, "Waiting for xal-server to index %s\n", device->data_dir);
		}
		if (waited >= XAL_REINDEX_TIMEOUT_MS) {
			fprintf(stderr,
				"The index of %s stayed out of date; is xal-server running?\n",
				device->data_dir);
			return ESTALE;
		}
		nanosleep(&interval, NULL);
	}
}

static bool
_xal_current(struct fil_dev *device)
{
	return xal_get_seq_lock(device->xal) == device->xal_seq && !xal_is_dirty(device->xal);
}

static void
_dataset_free(struct fil_dev *device)
{
	free(device->files);
	free(device->paths);
	device->files = NULL;
	device->paths = NULL;
	device->n_files = 0;
}

struct fil_listing {
	struct fil_stats *stats;
	size_t paths_nbytes;
	size_t paths_cap;
	uint64_t files_cap;
};

static int
_add_file(struct fil_dev *device, struct fil_listing *l, const char *class_name,
	  const char *file_name, uint32_t label, uint64_t size, struct xal_inode *inode)
{
	size_t len = strlen(class_name) + 1 + strlen(file_name) + 1;

	if (l->paths_nbytes + len > l->paths_cap) {
		size_t cap = (l->paths_nbytes + len) * 2;
		char *paths = realloc(device->paths, cap);

		if (!paths) {
			return errno;
		}
		device->paths = paths;
		l->paths_cap = cap;
	}
	if (device->n_files == l->files_cap) {
		uint64_t cap = l->files_cap ? l->files_cap * 2 : 1024;
		struct fil_file *files = realloc(device->files, cap * sizeof(*files));

		if (!files) {
			return errno;
		}
		device->files = files;
		l->files_cap = cap;
	}

	snprintf(device->paths + l->paths_nbytes, len, "%s/%s", class_name, file_name);
	device->files[device->n_files++] = (struct fil_file){
		.inode = inode, .path = l->paths_nbytes, .label = label, .seq = device->xal_seq};
	l->paths_nbytes += len;

	l->stats->n_files++;
	l->stats->avg_file_size += size;
	if (size > l->stats->max_file_size) {
		l->stats->max_file_size = size;
	}
	return 0;
}

// The list refers to the index only by inode, so a re-index does not invalidate it
static int
_read_dataset(struct fil_dev *device, struct fil_stats *stats)
{
	struct fil_listing l = {.stats = stats};
	struct xal_dentries classes;
	struct xal_inode *root;
	int err;

	err = _xal_settle(device);
	if (err) {
		return err;
	}

	err = _find_data_dir(device->xal, device->data_dir, &root);
	if (err) {
		return err;
	}
	classes = root->content.dentries;

	for (uint32_t i = 0; i < classes.count; i++) {
		struct xal_inode *class_dir = xal_inode_at(device->xal, classes.inodes_idx + i);
		struct xal_dentries dentries;

		if (!class_dir) {
			return ESTALE;
		}
		if (!xal_inode_is_dir(class_dir)) {
			fprintf(stderr, "Not a class directory: %s/%s\n", device->data_dir,
				class_dir->name);
			return EINVAL;
		}
		dentries = class_dir->content.dentries;

		for (uint32_t j = 0; j < dentries.count; j++) {
			struct xal_inode *inode =
				xal_inode_at(device->xal, dentries.inodes_idx + j);

			if (!inode) {
				return ESTALE;
			}
			if (!xal_inode_is_file(inode)) {
				fprintf(stderr, "Not a regular file: %s/%s/%s\n", device->data_dir,
					class_dir->name, inode->name);
				return EINVAL;
			}
			err = _add_file(device, &l, class_dir->name, inode->name, i, inode->size,
					inode);
			if (err) {
				return err;
			}
		}
	}

	return _xal_current(device) ? 0 : ESTALE;
}

static int
_read_index(struct fil_iter *iter, struct fil_dev *device, struct fil_stats *stats)
{
	int err;

	err = _xal_attach(iter->opts->xal_shm, &device->xal);
	if (err) {
		return err;
	}
	device->fs_blksize = xal_get_sb_blocksize(device->xal);

	for (uint32_t attempt = 0; attempt < FIL_READ_ATTEMPTS; attempt++) {
		memset(stats, 0, sizeof(*stats));
		_dataset_free(device);

		err = _read_dataset(device, stats);
		if (err != ESTALE) {
			break;
		}
	}
	return err;
}

static int
_skip_dots(const struct dirent *entry)
{
	return strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..");
}

// As the index sorts, so the labels match
static int
_by_name(const struct dirent **a, const struct dirent **b)
{
	return strcmp((*a)->d_name, (*b)->d_name);
}

static void
_free_dirents(struct dirent **entries, int n)
{
	for (int i = 0; i < n; i++) {
		free(entries[i]);
	}
	free(entries);
}

static int
_list_class(struct fil_dev *device, struct fil_listing *l, int data_fd, const char *class_name,
	    uint32_t label)
{
	struct dirent **files;
	int class_fd, n, err = 0;

	class_fd = openat(data_fd, class_name, O_RDONLY | O_DIRECTORY);
	if (class_fd == -1) {
		err = errno;
		fprintf(stderr, "%s %s/%s\n",
			err == ENOTDIR ? "Not a class directory:" : "Could not open",
			device->data_dir, class_name);
		return err == ENOTDIR ? EINVAL : err;
	}

	n = scandirat(class_fd, ".", &files, _skip_dots, _by_name);
	if (n < 0) {
		err = errno;
		fprintf(stderr, "Could not list %s/%s: %d\n", device->data_dir, class_name, err);
		close(class_fd);
		return err;
	}

	for (int j = 0; j < n && !err; j++) {
		struct stat st;

		if (fstatat(class_fd, files[j]->d_name, &st, AT_SYMLINK_NOFOLLOW) == -1) {
			err = errno;
			fprintf(stderr, "Could not stat %s/%s/%s: %d\n", device->data_dir,
				class_name, files[j]->d_name, err);
		} else if (!S_ISREG(st.st_mode)) {
			fprintf(stderr, "Not a regular file: %s/%s/%s\n", device->data_dir,
				class_name, files[j]->d_name);
			err = EINVAL;
		} else {
			err = _add_file(device, l, class_name, files[j]->d_name, label, st.st_size,
					NULL);
		}
	}

	_free_dirents(files, n);
	close(class_fd);
	return err;
}

static int
_list_dataset(struct fil_dev *device, struct fil_stats *stats)
{
	struct fil_listing l = {.stats = stats};
	struct dirent **classes;
	struct statvfs sv;
	int data_fd, n, err = 0;

	data_fd = open(device->data_dir, O_RDONLY | O_DIRECTORY);
	if (data_fd == -1) {
		err = errno;
		fprintf(stderr, "Could not open %s: %d\n", device->data_dir, err);
		return err;
	}
	if (fstatvfs(data_fd, &sv) == -1) {
		err = errno;
		fprintf(stderr, "Could not stat the file system of %s: %d\n", device->data_dir,
			err);
		close(data_fd);
		return err;
	}
	device->fs_blksize = sv.f_bsize;

	n = scandirat(data_fd, ".", &classes, _skip_dots, _by_name);
	if (n < 0) {
		err = errno;
		fprintf(stderr, "Could not list %s: %d\n", device->data_dir, err);
		close(data_fd);
		return err;
	}

	for (int i = 0; i < n && !err; i++) {
		err = _list_class(device, &l, data_fd, classes[i]->d_name, i);
	}

	_free_dirents(classes, n);
	close(data_fd);
	return err;
}

static int
_dataset_setup(struct fil_iter *iter, struct fil_dev *device)
{
	struct fil_stats stats = {0};
	uint64_t buffer_size;
	int err;

	device->data_dir = strdup(iter->opts->data_dir);
	if (!device->data_dir) {
		return errno;
	}

	if (iter->opts->xal_shm[0] == '\0') {
		err = _list_dataset(device, &stats);
	} else {
		err = _read_index(iter, device, &stats);
	}
	if (!err && !device->n_files) {
		fprintf(stderr, "No files in %s\n", device->data_dir);
		err = ENOENT;
	}
	if (err) {
		fprintf(stderr, "Dataset in %s cannot be read: %d\n", device->data_dir, err);
		_dataset_free(device);
		if (device->xal) {
			xal_close(device->xal);
			device->xal = NULL;
		}
		free(device->data_dir);
		device->data_dir = NULL;
		return err;
	}

	iter->stats->n_files += stats.n_files;
	iter->stats->avg_file_size += stats.avg_file_size;
	if (stats.max_file_size > iter->stats->max_file_size) {
		iter->stats->max_file_size = stats.max_file_size;
	}

	// Align to the file system block size
	buffer_size = (1 + ((stats.max_file_size - 1) / device->fs_blksize)) * device->fs_blksize;
	if (buffer_size > iter->buffer_size) {
		iter->buffer_size = buffer_size;
	}
	return 0;
}

int
fil_file_path(struct fil_dev *device, struct fil_file *file, char *path, size_t len)
{
	if (snprintf(path, len, "%s/%s", device->data_dir, device->paths + file->path) >=
	    (int)len) {
		return ENAMETOOLONG;
	}
	return 0;
}

int
fil_file_inode(struct fil_iter *iter, struct fil_dev *device, struct fil_file *file,
	       struct xal_inode **inode)
{
	char path[PATH_MAX];
	struct xal_inode *found;
	int err;

	if (file->seq == device->xal_seq) {
		*inode = file->inode;
		return 0;
	}

	err = fil_file_path(device, file, path, sizeof(path));
	if (err) {
		return err;
	}

	err = xal_get_inode(device->xal, path, &found);
	if (!_xal_current(device)) {
		return ESTALE;
	}
	if (err) {
		fprintf(stderr, "%s is gone from the index: %d\n", path, err);
		return -err;
	}
	if (!xal_inode_is_file(found)) {
		fprintf(stderr, "Not a regular file: %s\n", path);
		return EINVAL;
	}
	if (found->size > iter->buffer_size) {
		fprintf(stderr, "%s grew to %lu bytes, past its buffer of %lu\n", path,
			found->size, iter->buffer_size);
		return EFBIG;
	}

	file->inode = found;
	file->seq = device->xal_seq;
	*inode = found;
	return 0;
}

static int
_create_order(struct fil_iter *iter)
{
	uint64_t n_entries = iter->devs[0]->n_files;
	uint64_t *order;

	order = malloc(sizeof(*order) * n_entries);
	if (!order) {
		fprintf(stderr, "Could not allocate the order of the dataset: %d\n", errno);
		return errno;
	}
	for (uint64_t k = 0; k < n_entries; k++) {
		order[k] = k;
	}

	iter->data->order = order;
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
							     device->fs_blksize,
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
		if (device->xal) {
			xal_close(device->xal);
		}
		xnvme_dev_close(device->dev);
#ifdef FIL_CUDA_ENABLED
		cuFileDriverClose();
#endif
		if (device->file_io) {
			free(device->file_io->buffer);
			free(device->file_io);
		}
		free(device->buffers);
		_dataset_free(device);
		free(device->data_dir);
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
		free(iter->data->order);
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

	if (opts->xal_shm[0] == '\0' && strncmp(opts->backend, "aisio-", strlen("aisio-")) == 0) {
		fprintf(stderr, "xal_shm is required for the aisio backends\n");
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

		err = _dataset_setup(_iter, device);
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

	err = _create_order(_iter);
	if (err) {
		fil_term(_iter);
		return err;
	}

	FIL_SHUFFLE(_iter->data->order, uint64_t, _iter->data->n_entries, uint64_t);

	(*iter) = _iter;

	return 0;
}

// A re-index racing the reads can give their blocks to other files, so the batch is checked
// once they completed, and read again
static int
_read_batch(struct fil_iter *iter)
{
	uint64_t index = iter->data->index, bytes = iter->stats->bytes, io = iter->stats->io;
	uint64_t bufs[iter->n_devs];
	int err;

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		bufs[i] = iter->devs[i]->buf;
	}

	for (uint32_t attempt = 0; attempt < FIL_READ_ATTEMPTS; attempt++) {
		bool current = true;

		for (uint32_t i = 0; i < iter->n_devs; i++) {
			err = _xal_settle(iter->devs[i]);
			if (err) {
				return err;
			}
		}

		err = iter->io_fn(iter);
		if (err && err != ESTALE) {
			return err;
		}

		for (uint32_t i = 0; i < iter->n_devs; i++) {
			current = current && _xal_current(iter->devs[i]);
		}
		if (!err && current) {
			return 0;
		}

		iter->data->index = index;
		iter->stats->bytes = bytes;
		iter->stats->io = io;
		for (uint32_t i = 0; i < iter->n_devs; i++) {
			iter->devs[i]->buf = bufs[i];
		}
	}

	fprintf(stderr, "The index changed during %u reads of the batch\n", FIL_READ_ATTEMPTS);
	return ESTALE;
}

int
fil_next(struct fil_iter *iter, struct fil_output **output)
{
	int err;

	if (iter->data->index >= iter->data->n_entries) {
		iter->data->index = 0;
		FIL_SHUFFLE(iter->data->order, uint64_t, iter->data->n_entries, uint64_t);
	}

	err = iter->type == FIL_FILE ? iter->io_fn(iter) : _read_batch(iter);
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
				.nsid = 1,
				.iosize = 4096,
				.gpu_nqueues = 1,
				.queue_depth = 128,
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
