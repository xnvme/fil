# FIL: File Iterator Library

FIL is a library for iterating batch-wise over the files of a mounted file system,
reading them directly from NVMe devices. It supports multiple backends for CPU and
GPU workloads, enabling high-throughput data loading for deep learning and HPC use
cases.

## Dependencies

- [xNVMe](https://xnvme.io/) (`main` branch), with `homi` and `qublk` for the `aisio`
  backends
- [xal](https://github.com/xnvme/xal) 0.3.4, with `xal-server` for the `aisio` backends
- CUDA with cuFile, optional: without it the `aisio-gpu` and `cufile` backends and
  `--copy-to-gpu` are not built, and the others are. The `cuda` build option
  (`auto`, `enabled`, `disabled`) chooses.

## Building

```
make        # configure, build, and install
make clean  # remove build directory
```

A debug build can be configured with `make config-debug` followed by `make build install`.

## How it fits together

The `aisio` backends read the files of a mounted file system straight from the NVMe
device:

1. `homi` holds the controller and shares it between processes.
2. `qublk` serves the namespace through `homi` as `/dev/ublkb<N>`, which is mounted.
3. `xal-server` publishes an index of the file system, with the extents of every file.
4. FIL looks up the extents in the index and reads them over queues of its own from
   `homi`.

The `posix` and `cufile` backends read through the file system and need none of this.

For example, with the controller at `0000:03:00.0`:

```
homi serve 0000:03:00.0 --be upcie --homi-id 1
qublk serve 0000:03:00.0 --be upcie --homi-id 1 --dev-id 1
mount /dev/ublkb1 /mnt/datasets
xal-server --config /usr/local/etc/xal/xal-server.conf
```

where `xal-server.conf` publishes the index as `/xal_dev0`:

```toml
devices = [
  { uri = "/dev/ublkb1", shm_name = "/xal_dev0", mountpoint = "/mnt/datasets" },
]
```

The `homi` host heap (`--host_heap_size`) must hold the buffers of `aisio-cpu`:
`--batch-size` of them, each the size of the largest file.

When `xal-server` re-indexes (with a `watchmode`), FIL waits and reads any batch the
re-index raced again. A re-index walks the whole file system, so keep files that
change often elsewhere. Without a `watchmode`, the index is valid only while the file
system does not change. Sync the file system after writing the dataset.

## Directory structure

FIL expects data organized in two levels of subdirectories under a data directory:

```
<data_dir>/class_a/file0
<data_dir>/class_a/file1
<data_dir>/class_b/file2
<data_dir>/class_b/file3
...
```

The subdirectory names are used as class labels. The `data_dir` is an absolute path on
the mounted file system.

## CLI

The `filperf` CLI runs benchmarks and reports throughput metrics. Progress is printed
as CSV every second: elapsed time, batches completed, IOPS, and MiB/s.

```
filperf [<device-uri>] --data-dir <path> [options]
```

### Options

| Option | Default | Description |
|---|---|---|
| `--data-dir <path>` | _(required)_ | Absolute path to the directory containing class subdirectories |
| `--xal-shm <name>` | _(none)_ | Name of the `xal-server` index; required for the `aisio` backends |
| `--nsid <n>` | `1` | Namespace |
| `--backend <name>` | `aisio-cpu` | I/O backend: `aisio-cpu`, `aisio-gpu`, `aisio-p2p`, `posix`, `cufile`, `cufile-stream` |
| `--homi-id <n>` | _(none)_ | Id of the `homi` server holding the device; required for the `aisio` backends |
| `--batch-size <n>` | `1` | Number of files per batch |
| `--batches <n>` | `1` | Number of batches to read |
| `--iosize <n>` | `4096` | Number of bytes per I/O (`aisio` backends only) |
| `--queue-depth <n>` | `128` | NVMe queue depth (`aisio` backends only) |
| `--gpu-nqueues <n>` | `1` | Number of GPU queues (`aisio-gpu` only) |
| `--warmup <n>` | `0` | Un-timed batches to run before starting the measurement window |
| `--buffered` | off | Disable `O_DIRECT` when using `posix` backend |
| `--copy-to-gpu` | off | Copy each file from host to device memory after reading (`aisio-cpu` and `posix` only) |
| `--summary` | off | Print I/O and dataset statistics after completion |
| `--help` | | Print usage |

`posix` and `cufile` need no device uri.

### Examples

`aisio-cpu` reads directly from NVMe to CPU memory using xNVMe, through `homi`:

```
filperf 0000:03:00.0 --homi-id 1 --xal-shm /xal_dev0 --data-dir /mnt/datasets/imagenetish \
  --backend aisio-cpu --batches 10 --batch-size 1024 --summary
```

`posix` and `cufile` read through the mounted file system.
`cufile` additionally requires CUDA with the cuFile library, and `cufile-stream` is the
same backend with every read ordered on a CUDA stream through `cuFileReadAsync`.
Direct NVMe-to-GPU (GPUDirect Storage) transfers need the kernel-side GDS driver
(`nvidia-fs`); without it, cuFile falls back to a compatibility path that stages
through host memory.
For these file-level backends, IOPS equals File/s since each file is a single I/O.

```
filperf --data-dir /mnt/datasets/imagenetish \
  --backend posix --batches 10 --batch-size 1024 --summary
```

## API

Initialize the iterator with `fil_init()`, call `fil_next()` to retrieve successive
batches, and clean up with `fil_term()`. Each batch is described by a `fil_output`
struct containing buffers, their lengths, and class labels derived from the
subdirectory layout. I/O statistics accumulated across batches are available via
`fil_get_stats()`.

`aisio-gpu`, `aisio-p2p` and `cufile` return buffers in GPU memory. `aisio-cpu`
and `posix` return host memory unless `copy_to_gpu` is set, and then need no GPU.

See `include/libfil.h` for the full API reference.
