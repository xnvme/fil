// SPDX-FileCopyrightText: Samsung Electronics Co., Ltd
//
// SPDX-License-Identifier: BSD-3-Clause

#include <libxnvme.h>
#include <libxnvme_cuda.h>

/**
 * One CUDA block per queue; qdepth = blockDim.x threads per block.
 * Each block exclusively owns qps[blockIdx.x] — no cross-queue contention.
 * Commands are striped across queues: block b handles commands
 * [b*qdepth, b*qdepth+stride, b*qdepth+2*stride, ...] where
 * stride = n_queues * qdepth.
 * All threads in a block must call xnvme_cuda_cmd_io together on every
 * iteration because it contains __syncthreads() internally.
 */
__global__ static void
gpu_read_kernel(struct xnvme_cuda_queue **qps, struct xnvme_spec_cmd *cmds, uint32_t n_io)
{
	const uint32_t bid = blockIdx.x;
	const uint32_t tid = threadIdx.x;
	const uint32_t qdepth = blockDim.x;
	const uint32_t n_queues = gridDim.x;
	const uint32_t stride = n_queues * qdepth;

	for (uint32_t base = bid * qdepth; base < n_io; base += stride) {
		uint32_t batch = min(qdepth, n_io - base);
		// Threads with tid >= batch pass cmd[0] as a dummy;
		// xnvme_cuda_cmd_io is a no-op for them
		uint32_t cmd_idx = (base + tid < n_io) ? (base + tid) : 0;

		xnvme_cuda_cmd_io(qps[bid], &cmds[cmd_idx], tid, batch);
	}
}

extern "C" void
gpu_io_launch(struct xnvme_cuda_queue **queues_dev, uint32_t n_queues,
	      struct xnvme_spec_cmd *cmds_dev, uint32_t n_io, uint32_t queue_depth)
{
	gpu_read_kernel<<<n_queues, queue_depth>>>(queues_dev, cmds_dev, n_io);
}
