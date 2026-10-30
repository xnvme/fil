#ifndef __FIL_ITER_H
#define __FIL_ITER_H

#include <libfil.h>
#include <libxnvme.h>
#include <fil_io.h>
#include <stdint.h>

enum fil_type { FIL_GPU, FIL_CPU, FIL_FILE, FIL_P2P, FIL_CUFILE, FIL_OPENDS };

enum fil_mode { FIL_SYNC, FIL_STREAM };

struct fil_time;

struct fil_dev {
	struct xnvme_dev *dev;
	struct xnvme_queue *queue;
	struct xnvme_cuda_queue **cuda_queues;
	struct xnvme_cuda_queue **cuda_queues_dev;
	struct fil_gpu_io gpu_io;
	struct xal *xal;
	struct fil_file *files; ///< Sorted by class directory, then by name
	struct fil_file_io *file_io;
	int xal_seq; ///< Sequence lock of the batch in flight
	char *data_dir;
	char *paths; ///< Paths relative to data_dir
	uint64_t n_files;
	uint32_t fs_blksize;
	uint32_t io_errors; ///< Reads that failed in the current batch (CPU/P2P path)
	void **buffers;
	void **gpu_buffers; ///< Per-buffer GPU destinations for the aisio-cpu copy_to_gpu path
	uint64_t buf;
	uint32_t n_buffers;
	uint32_t nsid;
};

struct fil_iter {
	struct fil_dev **devs;
	struct fil_data *data;
	struct fil_stats *stats;
	struct fil_opts *opts;
	struct fil_output *output;
	struct fil_cufile_io *cufile_io;
	struct fil_opends_io *opends_io;
	int (*io_fn)(struct fil_iter *iter);
	struct fil_time *time;
	uint64_t buffer_size;
	uint32_t n_devs;
	enum fil_type type;
	enum fil_mode mode;
};

/** Write the absolute path of 'file' to 'path' */
int
fil_file_path(struct fil_dev *device, struct fil_file *file, char *path, size_t len);

/**
 * Where the index has 'file' now, looked up again by path after a re-index. Fails with
 * ESTALE, ENOENT if the file is gone, or EFBIG if it outgrew the buffers.
 */
int
fil_file_inode(struct fil_iter *iter, struct fil_dev *device, struct fil_file *file,
	       struct xal_inode **inode);

#endif
