#!/usr/bin/env python3
"""
Populate the file system with a dataset for fil
===============================================

Writes class directories under config.fil.testing.mountpoint/data_dir using
cijoe/auxiliary/fil_dataset.py of the fil source on the target, covering
sub-block, block-aligned and block-plus-one file sizes.

Two more files are grown by interleaved direct appends, so that each spans
many extents and the multi-extent read path gets exercised; a buffered tail
leaves each with a partial last block. The step fails if they do not end up
fragmented.

Finally, the file system is mounted again read-only, so that all meta-data is
on the device when fil parses it from there.

Retargetable: True
------------------
"""
import logging as log
from argparse import Namespace

from cijoe.core.command import Cijoe


def main(args: Namespace, cijoe: Cijoe):
    """Entry-point of the cijoe-script"""

    testing = cijoe.getconf("fil.testing")
    dataset = f"{cijoe.getconf('fil.source')}/cijoe/auxiliary/fil_dataset.py"
    dev = testing["dev"]
    mountpoint = testing["mountpoint"]
    data_dir = f"{mountpoint}/{testing['data_dir']}"
    frag_dir = f"{data_dir}/class_0"

    err, _ = cijoe.run(
        f"python3 {dataset} populate --mnt {mountpoint} --data-dir {testing['data_dir']}"
    )
    if err:
        return err

    err, _ = cijoe.run(
        f"touch {frag_dir}/frag_a {frag_dir}/frag_b && "
        "for i in $(seq 16); do for f in frag_a frag_b; do "
        f"dd if=/dev/urandom of={frag_dir}/$f bs=4096 count=1 "
        "oflag=direct,append conv=notrunc status=none || exit 1; "
        "done; done"
    )
    if err:
        return err

    for name in ["frag_a", "frag_b"]:
        err, _ = cijoe.run(f"head -c 1234 /dev/urandom >> {frag_dir}/{name}")
        if err:
            return err

        err, state = cijoe.run(f"filefrag {frag_dir}/{name}")
        if err:
            return err
        extents = int(state.output().split(":")[-1].split()[0])
        if extents < 2:
            log.error(f"{name}: {extents} extent(s); expected it to be fragmented")
            return 1

    for cmd in [
        f"find {data_dir} -type f | wc -l",
        f"umount {mountpoint}",
        f"mount -o ro {dev} {mountpoint}",
    ]:
        err, _ = cijoe.run(cmd)
        if err:
            return err

    return 0
