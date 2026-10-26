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
#include <fil_iter.h>
#include <fil_time.h>
#include <fil_util.h>

#include <libxal.h>

#include <cuda_runtime.h>
#include <opends.h>

struct fil_opends_io {
	opends_handle_t *handles;
	int *fds;
	size_t *expected;
	ssize_t *actual;
	cudaStream_t *streams;
	opends_async_future_t *futures;
	void **bufs;
};

int
fil_opends_setup(void)
{
	opends_error_t derr;

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

	io->handles = malloc(sizeof(opends_handle_t) * iter->opts->batch_size);
	if (!io->handles) {
		err = errno;
		fprintf(stderr, "Could not allocate OpenDS handles: %d\n", err);
		return err;
	}

	io->fds = malloc(sizeof(int) * iter->opts->batch_size);
	if (!io->fds) {
		err = errno;
		fprintf(stderr, "Could not allocate array of fds: %d\n", err);
		return err;
	}

	io->expected = malloc(sizeof(size_t) * iter->opts->batch_size);
	if (!io->expected) {
		err = errno;
		fprintf(stderr, "Could not allocate array of expected values: %d\n", err);
		return err;
	}

	io->actual = malloc(sizeof(ssize_t) * iter->opts->batch_size);
	if (!io->actual) {
		err = errno;
		fprintf(stderr, "Could not allocate array of actual values: %d\n", err);
		return err;
	}

	if (iter->mode == FIL_ASYNC) {
		io->bufs = malloc(sizeof(void *) * iter->opts->batch_size);
		if (!io->bufs) {
			err = errno;
			fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
			return err;
		}

		io->futures = malloc(sizeof(opends_async_future_t) * iter->opts->batch_size);
		if (!io->futures) {
			err = errno;
			fprintf(stderr, "Could not allocate array of futures: %d\n", err);
			return err;
		}

		return 0;
	}

	io->streams = calloc(iter->opts->batch_size, sizeof(cudaStream_t));
	if (!io->streams) {
		err = errno;
		fprintf(stderr, "Could not allocate array of CUDA Streams: %d\n", err);
		return err;
	}

	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
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
		free(io->bufs);
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

/* Reads the whole file through the handle and releases it. */
ssize_t
fil_opends_file_read(void *fh, void *buf, size_t nbytes, const char *path)
{
	ssize_t bytes_read;

	bytes_read = opends_sync_read(fh, buf, nbytes, 0, 0);
	opends_handle_deregister(fh);
	if (bytes_read < 0) {
		fprintf(stderr, "Could not read %s, err: %s\n", path,
			opends_op_status_error((opends_op_error_t)(-bytes_read)));
	}

	return bytes_read;
}

int
fil_opends_stream_submit(struct fil_iter *iter)
{
	struct xal_inode *dir, *file;
	struct timespec start, end;
	struct fil_opends_io *io = iter->opends_io;
	opends_error_t derr;
	uint32_t buf_id, dev_id, nsub = 0;
	void *buffer;
	char *prefix, *path;
	off_t offset = 0;
	int err = 0;
	int flags = O_RDONLY | O_DIRECT;

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
		strcat(path, dir->name);
		strcat(path, "/");
		strcat(path, file->name);

		io->fds[i] = open(path, flags);
		if (io->fds[i] == -1) {
			err = errno;
			fprintf(stderr, "Could not open %s, err: %d\n", path, err);
			goto teardown;
		}

		derr = opends_handle_register(&io->handles[i], io->fds[i]);
		if (derr.err != OPENDS_SUCCESS) {
			fprintf(stderr, "Could not register file, err: %s\n",
				opends_op_status_error(derr.err));
			close(io->fds[i]);
			err = derr.err;
			goto teardown;
		}

		io->expected[i] = file->size;
		io->actual[i] = 0;
		nsub = i + 1;

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
		if (io->actual[i] < 0) {
			fprintf(stderr, "Reading failed, err: %s\n",
				opends_op_status_error((opends_op_error_t)(-io->actual[i])));
			err = EIO;
			goto teardown;
		}
		if ((size_t)io->actual[i] != io->expected[i]) {
			fprintf(stderr, "Could not read entire file, expected: %lu, actual: %ld\n",
				io->expected[i], io->actual[i]);
			err = EIO;
			goto teardown;
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += ELAPSED(start, end);

teardown:
	for (uint32_t i = 0; i < nsub; i++) {
		opends_handle_deregister(io->handles[i]);
		close(io->fds[i]);
	}
	return err;
}

int
fil_opends_async_submit(struct fil_iter *iter)
{
	struct xal_inode *dir, *file;
	struct timespec start, end;
	struct fil_opends_io *io = iter->opends_io;
	opends_error_t derr;
	uint32_t buf_id, dev_id, nreg = 0, nsub = 0;
	char *prefix, *path;
	int err = 0;
	int flags = O_RDONLY | O_DIRECT;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		dev_id = i % iter->n_devs;
		struct fil_dev *device = iter->devs[dev_id];
		buf_id = device->buf++ % device->n_buffers;
		io->bufs[i] = device->buffers[buf_id];
		prefix = device->file_io->prefix;
		path = device->file_io->path;

		file = fil_next_file(iter, device, dev_id, buf_id, &dir);
		iter->stats->io++;

		memcpy(path, prefix, strlen(prefix) + 1);
		strcat(path, "/");
		strcat(path, dir->name);
		strcat(path, "/");
		strcat(path, file->name);

		io->fds[i] = open(path, flags);
		if (io->fds[i] == -1) {
			err = errno;
			fprintf(stderr, "Could not open %s, err: %d\n", path, err);
			break;
		}

		derr = opends_handle_register(&io->handles[i], io->fds[i]);
		if (derr.err != OPENDS_SUCCESS) {
			fprintf(stderr, "Could not register file, err: %s\n",
				opends_op_status_error(derr.err));
			close(io->fds[i]);
			err = derr.err;
			break;
		}

		io->expected[i] = file->size;
		nreg = i + 1;

		derr = opends_async_read(io->handles[i], io->bufs[i], io->expected[i], 0, 0,
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
		ssize_t n = opends_async_await(&io->futures[i]);

		if (n < 0) {
			fprintf(stderr, "Reading failed, err: %s\n",
				opends_op_status_error((opends_op_error_t)-n));
			err = EIO;
		} else if ((size_t)n != io->expected[i]) {
			fprintf(stderr, "Could not read entire file, expected: %lu, actual: %zd\n",
				io->expected[i], n);
			err = EIO;
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += ELAPSED(start, end);

	/* Deregister only after the await loop: a handle with I/O in flight on it
	 * must not be deregistered. */
	for (uint32_t i = 0; i < nreg; i++) {
		opends_handle_deregister(io->handles[i]);
		close(io->fds[i]);
	}
	return err;
}
