#include <errno.h>
#include <stdio.h>

#include <libfil.h>
#include <fil_io.h>
#include <fil_util.h>

int
fil_opends_setup(void)
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

ssize_t
fil_opends_file_read(void *FIL_UNUSED(fh), void *FIL_UNUSED(buf), size_t FIL_UNUSED(nbytes),
		     const char *FIL_UNUSED(path))
{
	return -ENOSYS;
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
