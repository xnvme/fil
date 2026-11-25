#ifndef __FIL_ITER_H
#define __FIL_ITER_H

#include <libfil.h>
#include <libxnvme.h>
#include <fil_io.h>
#ifdef FIL_CUDA_ENABLED
#include <cuda.h>
#endif
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>

enum fil_type { FIL_GPU, FIL_CPU, FIL_FILE, FIL_P2P, FIL_CUFILE, FIL_OPENDS };

enum fil_mode { FIL_SYNC, FIL_STREAM, FIL_ASYNC };

struct fil_time;

struct fil_dev {
	struct xnvme_dev *dev;
	struct xnvme_cuda_queue **cuda_queues;
	struct xnvme_cuda_queue **cuda_queues_dev;
	struct fil_gpu_io gpu_io;
	struct xal *xal;
	struct fil_file *files; ///< Sorted by class directory, then by name
	int xal_seq;            ///< Sequence lock of the batch in flight
	char *data_dir;
	char *paths; ///< Paths relative to data_dir
	uint64_t n_files;
	uint32_t fs_blksize;
	void **buffers;
	void **gpu_buffers; ///< Per-buffer GPU destinations for the aisio-cpu copy_to_gpu path
	uint32_t n_buffers;
	uint32_t nsid;
};

/**
 * A thread reading the slots [first, first + n) of each batch. The first worker reads on the
 * thread that calls fil_next(); the others run for the life of the iterator and wait for
 * each batch.
 */
struct fil_worker {
	struct fil_iter *iter;
	struct xnvme_queue *queue; ///< aisio-cpu and aisio-p2p
	struct fil_stats stats;    ///< Of the batch in flight
	char path[PATH_MAX];
	void *bounce; ///< Host buffer the copy_to_gpu reads land in
#ifdef FIL_CUDA_ENABLED
	cudaStream_t stream; ///< The copy_to_gpu copies, off the shared default stream
#endif
	pthread_t thread;
	bool running;       ///< 'thread' was started and is to be joined
	uint64_t base;      ///< Position in the order of the batch's first file
	uint32_t first;     ///< First slot of the batch this worker reads
	uint32_t n;         ///< Number of slots it reads
	uint32_t io_errors; ///< Reads that failed in the batch in flight (CPU/P2P path)
	int init_err;       ///< Setting the thread up failed; fil_init() reports it
	int err;
};

/** Hands each batch to the worker threads and waits for them */
struct fil_pool {
	pthread_mutex_t lock;
	pthread_cond_t start; ///< A batch is ready, or the pool stops
	pthread_cond_t done;  ///< The last worker set up, or finished its slots
	uint64_t gen;         ///< Batches handed out so far
	uint32_t pending;     ///< Threads still reading the batch
	uint32_t ready;       ///< Threads that finished setting up
	bool stop;
	bool init; ///< 'lock' and the conditions were initialized
};

struct fil_iter {
	struct fil_dev **devs;
	struct fil_data *data;
	struct fil_stats *stats;
	struct fil_opts *opts;
	struct fil_output *output;
	struct fil_cufile_io *cufile_io;
	struct fil_opends_io *opends_io;
	struct fil_worker *workers;
	struct fil_pool pool;
#ifdef FIL_CUDA_ENABLED
	CUcontext cu_ctx; ///< Made current on the worker threads; NULL if they make no CUDA calls
#endif
	int (*io_fn)(struct fil_worker *w);
	struct fil_time *time;
	uint64_t buffer_size;
	uint32_t n_devs;
	uint32_t n_workers;
	pid_t pid; ///< The process the worker threads run in; a fork leaves them behind
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
