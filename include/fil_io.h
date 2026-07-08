#ifndef __FIL_IO_H
#define __FIL_IO_H

#include <cuda_runtime.h>
#include <cufile.h>
#include <limits.h>
#include <stdint.h>

struct fil_entry {
	uint64_t dir;
	uint64_t file;
};

struct fil_data {
	struct fil_entry *entries;
	uint64_t n_entries;
	uint64_t index;
};

struct fil_file_io {
	char prefix[PATH_MAX];
	char path[PATH_MAX];
	void *buffer;
};

struct fil_cufile_io {
	CUfileDescr_t *descr;
	CUfileHandle_t *handle;
	size_t *expected;
	ssize_t *actual;
	cudaStream_t *streams;
};

struct fil_gpu_io {
	struct xnvme_spec_cmd *cmds_host;
	struct xnvme_spec_cmd *cmds_dev;
	uint64_t *prp1_base;
	uint32_t n_io;
};

struct xnvme_cuda_queue;

void
gpu_io_launch(struct xnvme_cuda_queue **queues_dev, uint32_t n_queues,
	      struct xnvme_spec_cmd *cmds_dev, uint32_t n_io, uint32_t queue_depth);

struct xnvme_cmd_ctx;

void
fil_io_cb(struct xnvme_cmd_ctx *ctx, void *cb_arg);

int
fil_cpu_submit(struct fil_iter *iter);

int
fil_gpu_submit(struct fil_iter *iter);

int
fil_file_submit(struct fil_iter *iter);

int
fil_cufile_async_submit(struct fil_iter *iter);

#endif
