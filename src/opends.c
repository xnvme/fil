#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <libfil.h>
#include <fil_io.h>
#include <fil_opends.h>
#include <fil_iter.h>
#include <fil_time.h>
#include <fil_util.h>

#ifdef FIL_OPENDS_ENABLED
#include <cuda_runtime.h>
#include <opends.h>

struct fil_opends_io {
	opends_handle_t *handles;
	int *fds;
	size_t *expected;
	ssize_t *actual;
	cudaStream_t *streams;
	opends_async_future_t *futures;
};

/* OpenDS is configured through the environment, which it reads as the driver opens */
int
fil_opends_setup(const struct fil_opts *opts)
{
	opends_error_t derr;
	char homi_id[16];

	snprintf(homi_id, sizeof(homi_id), "%u", opts->homi_id);
	if (setenv("OPENDS_AISIO_HOMI_ID", homi_id, 1) ||
	    setenv("OPENDS_XAL_SHM", opts->xal_shm, 1)) {
		return errno;
	}

	derr = opends_driver_open();
	if (derr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "opends_driver_open: %s\n", opends_op_status_error(derr.err));
		return derr.err;
	}

	return 0;
}

int
fil_opends_buf_alloc(struct fil_iter *iter, void **buf)
{
	opends_error_t derr;
	int err;

	err = cudaMalloc(buf, iter->buffer_size);
	if (err) {
		return err;
	}
	derr = opends_buf_register(*buf, iter->buffer_size, 0);
	if (derr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "opends_buf_register: %s\n", opends_op_status_error(derr.err));
		return derr.err;
	}

	return 0;
}

void
fil_opends_buf_free(void *buf)
{
	opends_buf_deregister(buf);
	cudaFree(buf);
}

int
fil_opends_io_alloc(struct fil_iter *iter)
{
	uint32_t n = iter->opts->batch_size;
	struct fil_opends_io *io;
	int err;

	if (iter->mode == FIL_SYNC) {
		return 0;
	}

	io = calloc(1, sizeof(*io));
	if (!io) {
		err = errno;
		fprintf(stderr, "Could not allocate OpenDS IO struct: %d\n", err);
		return err;
	}
	iter->opends_io = io;

	io->handles = malloc(sizeof(opends_handle_t) * n);
	io->fds = malloc(sizeof(int) * n);
	io->expected = malloc(sizeof(size_t) * n);
	io->actual = malloc(sizeof(ssize_t) * n);
	io->futures = malloc(sizeof(opends_async_future_t) * n);
	if (!io->handles || !io->fds || !io->expected || !io->actual || !io->futures) {
		err = errno;
		fprintf(stderr, "Could not allocate OpenDS IO arrays: %d\n", err);
		return err;
	}

	if (iter->mode == FIL_ASYNC) {
		return 0;
	}

	io->streams = calloc(n, sizeof(cudaStream_t));
	if (!io->streams) {
		err = errno;
		fprintf(stderr, "Could not allocate array of CUDA Streams: %d\n", err);
		return err;
	}

	for (uint32_t i = 0; i < n; i++) {
		opends_error_t derr;

		err = cudaStreamCreateWithFlags(&io->streams[i], cudaStreamNonBlocking);
		if (err) {
			fprintf(stderr, "Could not setup CUDA Stream, err: %d\n", err);
			return err;
		}
		derr = opends_stream_register(io->streams[i], 0);
		if (derr.err != OPENDS_SUCCESS) {
			fprintf(stderr, "opends_stream_register: %s\n",
				opends_op_status_error(derr.err));
			return derr.err;
		}
	}

	return 0;
}

/* The streams are registered with the driver, so they go before it closes. */
void
fil_opends_term(struct fil_iter *iter)
{
	struct fil_opends_io *io = iter->opends_io;

	if (io) {
		free(io->handles);
		free(io->fds);
		free(io->expected);
		free(io->actual);
		free(io->futures);
		if (io->streams) {
			for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
				if (!io->streams[i]) {
					continue;
				}
				opends_stream_deregister(io->streams[i]);
				cudaStreamDestroy(io->streams[i]);
			}
			free(io->streams);
		}
		free(io);
		iter->opends_io = NULL;
	}
	opends_driver_close();
}

int
fil_opends_file_register(int fd, void **fh)
{
	opends_error_t derr;

	derr = opends_handle_register(fh, fd);
	if (derr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "Could not register file, err: %s\n",
			opends_op_status_error(derr.err));
		return derr.err;
	}

	return 0;
}

static int
_check_read(ssize_t actual, size_t expected)
{
	if (actual < 0) {
		fprintf(stderr, "Reading failed, err: %s\n",
			opends_op_status_error((opends_op_error_t)(-actual)));
		return EIO;
	}
	if ((size_t)actual != expected) {
		fprintf(stderr, "Could not read entire file, expected: %lu, actual: %ld\n",
			expected, actual);
		return EIO;
	}

	return 0;
}

/* Reads the whole file through the handle and releases it. */
int
fil_opends_file_read(void *fh, void *buf, size_t nbytes, const char *path)
{
	ssize_t bytes_read;
	int err;

	bytes_read = opends_sync_read(fh, buf, nbytes, 0, 0);
	opends_handle_deregister(fh);

	err = _check_read(bytes_read, nbytes);
	if (err) {
		fprintf(stderr, "Could not read %s\n", path);
	}

	return err;
}

/* Opens the next file for read 'i' and registers it. */
static int
_prep_read(struct fil_iter *iter, uint32_t i, void **buffer)
{
	struct fil_opends_io *io = iter->opends_io;
	uint64_t size;
	int err;

	err = fil_open_next(iter, i, O_RDONLY | O_DIRECT, buffer, &io->fds[i], &size);
	if (err) {
		return err;
	}

	err = fil_opends_file_register(io->fds[i], &io->handles[i]);
	if (err) {
		close(io->fds[i]);
		return err;
	}

	io->expected[i] = size;
	io->actual[i] = 0;

	return 0;
}

static void
_release_reads(struct fil_opends_io *io, uint32_t n)
{
	for (uint32_t i = 0; i < n; i++) {
		opends_handle_deregister(io->handles[i]);
		close(io->fds[i]);
	}
}

int
fil_opends_stream_submit(struct fil_iter *iter)
{
	struct fil_opends_io *io = iter->opends_io;
	struct timespec start, end;
	opends_error_t derr;
	uint32_t nreg = 0;
	off_t offset = 0;
	void *buffer;
	int err = 0;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		err = _prep_read(iter, i, &buffer);
		if (err) {
			goto teardown;
		}
		nreg = i + 1;

		derr = opends_stream_read(io->handles[i], buffer, &io->expected[i], &offset,
					  &offset, &io->actual[i], io->streams[i]);
		if (derr.err != OPENDS_SUCCESS) {
			fprintf(stderr, "opends_stream_read failed, err: %s\n",
				opends_op_status_error(derr.err));
			err = derr.err;
			goto teardown;
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->prep_time += ELAPSED(start, end);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		err = cudaStreamSynchronize(io->streams[i]);
		if (err) {
			fprintf(stderr, "Could not synchronize CUDA Stream, err: %d\n", err);
			goto teardown;
		}
		err = _check_read(io->actual[i], io->expected[i]);
		if (err) {
			goto teardown;
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += ELAPSED(start, end);

teardown:
	_release_reads(io, nreg);
	return err;
}

int
fil_opends_async_submit(struct fil_iter *iter)
{
	struct fil_opends_io *io = iter->opends_io;
	struct timespec start, end;
	opends_error_t derr;
	uint32_t nreg = 0, nsub = 0;
	void *buffer;
	int err = 0;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		err = _prep_read(iter, i, &buffer);
		if (err) {
			break;
		}
		nreg = i + 1;

		derr = opends_async_read(io->handles[i], buffer, io->expected[i], 0, 0,
					 &io->futures[i]);
		if (derr.err != OPENDS_SUCCESS) {
			fprintf(stderr, "opends_async_read failed, err: %s\n",
				opends_op_status_error(derr.err));
			err = derr.err;
			break;
		}
		nsub = i + 1;
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->prep_time += ELAPSED(start, end);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < nsub; i++) {
		int failed = _check_read(opends_async_await(&io->futures[i]), io->expected[i]);

		if (failed) {
			err = failed;
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += ELAPSED(start, end);

	/* Deregister only after the await loop: a handle with I/O in flight on it
	 * must not be deregistered. */
	_release_reads(io, nreg);
	return err;
}

#else

int
fil_opends_setup(const struct fil_opts *FIL_UNUSED(opts))
{
	fprintf(stderr, "fil was built without the opends backend\n");
	return ENOSYS;
}

int
fil_opends_buf_alloc(struct fil_iter *FIL_UNUSED(iter), void **FIL_UNUSED(buf))
{
	return ENOSYS;
}

void
fil_opends_buf_free(void *FIL_UNUSED(buf))
{
}

int
fil_opends_io_alloc(struct fil_iter *FIL_UNUSED(iter))
{
	return ENOSYS;
}

void
fil_opends_term(struct fil_iter *FIL_UNUSED(iter))
{
}

int
fil_opends_file_register(int FIL_UNUSED(fd), void **FIL_UNUSED(fh))
{
	return ENOSYS;
}

int
fil_opends_file_read(void *FIL_UNUSED(fh), void *FIL_UNUSED(buf), size_t FIL_UNUSED(nbytes),
		     const char *FIL_UNUSED(path))
{
	return ENOSYS;
}

int
fil_opends_stream_submit(struct fil_iter *FIL_UNUSED(iter))
{
	return ENOSYS;
}

int
fil_opends_async_submit(struct fil_iter *FIL_UNUSED(iter))
{
	return ENOSYS;
}

#endif
