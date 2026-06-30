#ifndef __FIL_TIME_H
#define __FIL_TIME_H

#include <time.h>

#define ELAPSED(s, e) \
	((double)((e).tv_sec - (s).tv_sec) + (double)((e).tv_nsec - (s).tv_nsec) / 1e9)

#ifndef FIL_SUBTIME
#define FIL_SUBTIME 0
#endif

#if FIL_SUBTIME
#include <cuda_runtime.h>
#include <libfil.h>
#include <stdio.h>
#include <stdlib.h>

struct fil_time {
	struct timespec last;
	struct fil_stats *stats;
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_end;
};

static inline int
fil_time_alloc(struct fil_time **out, struct fil_stats *stats)
{
	int err;
	struct fil_time *t;

	t = malloc(sizeof(*t));
	if (!t) {
		fprintf(stderr, "Could not allocate fil_time\n");
		return -1;
	}
	t->stats = stats;
	err      = cudaEventCreate(&t->kernel_start);
	if (err) {
		fprintf(stderr, "cudaEventCreate(kernel_start): %d\n", err);
		free(t);
		return err;
	}
	err = cudaEventCreate(&t->kernel_end);
	if (err) {
		fprintf(stderr, "cudaEventCreate(kernel_end): %d\n", err);
		cudaEventDestroy(t->kernel_start);
		free(t);
		return err;
	}
	*out = t;
	return 0;
}

static inline void
fil_time_free(struct fil_time *t)
{
	if (!t)
		return;
	cudaEventDestroy(t->kernel_end);
	cudaEventDestroy(t->kernel_start);
	free(t);
}

static inline void
fil_time_begin(struct fil_time *t)
{
	clock_gettime(CLOCK_MONOTONIC_RAW, &t->last);
}

static inline void
fil_time_tick(struct fil_time *t, double *bucket)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC_RAW, &now);
	*bucket += (double)(now.tv_sec - t->last.tv_sec) +
		   (double)(now.tv_nsec - t->last.tv_nsec) / 1e9;
	t->last = now;
}

static inline void
fil_time_kernel_start(struct fil_time *t)
{
	cudaEventRecord(t->kernel_start, 0);
}

static inline void
fil_time_kernel_end(struct fil_time *t)
{
	cudaEventRecord(t->kernel_end, 0);
}

static inline void
fil_time_kernel_elapsed(struct fil_time *t)
{
	float ms = 0;
	cudaEventElapsedTime(&ms, t->kernel_start, t->kernel_end);
	t->stats->io_kernel_time += (double)ms / 1000.0;
}

#define FIL_TIME_ALLOC(iter)          fil_time_alloc(&(iter)->time, (iter)->stats)
#define FIL_TIME_FREE(iter)           fil_time_free((iter)->time)
#define FIL_TIME_BEGIN(iter)          fil_time_begin((iter)->time)
#define FIL_TIME_TICK(iter, field)    fil_time_tick((iter)->time, &(iter)->stats->field)
#define FIL_TIME_KERNEL_START(iter)   fil_time_kernel_start((iter)->time)
#define FIL_TIME_KERNEL_END(iter)     fil_time_kernel_end((iter)->time)
#define FIL_TIME_KERNEL_ELAPSED(iter) fil_time_kernel_elapsed((iter)->time)
#else
#define FIL_TIME_ALLOC(iter) 0
#define FIL_TIME_FREE(iter)
#define FIL_TIME_BEGIN(iter)
#define FIL_TIME_TICK(iter, field)
#define FIL_TIME_KERNEL_START(iter)
#define FIL_TIME_KERNEL_END(iter)
#define FIL_TIME_KERNEL_ELAPSED(iter)
#endif

#endif
