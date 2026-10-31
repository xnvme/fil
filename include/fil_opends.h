#ifndef __FIL_OPENDS_H
#define __FIL_OPENDS_H

#include <stddef.h>

struct fil_iter;
struct fil_opts;
struct fil_opends_io;

int
fil_opends_setup(const struct fil_opts *opts);

int
fil_opends_buf_alloc(struct fil_iter *iter, void **buf);

void
fil_opends_buf_free(void *buf);

int
fil_opends_io_alloc(struct fil_iter *iter);

void
fil_opends_term(struct fil_iter *iter);

int
fil_opends_file_register(int fd, void **fh);

int
fil_opends_file_read(void *fh, void *buf, size_t nbytes, const char *path);

int
fil_opends_stream_submit(struct fil_iter *iter);

int
fil_opends_async_submit(struct fil_iter *iter);

#endif
