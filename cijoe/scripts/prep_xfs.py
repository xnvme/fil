#!/usr/bin/env python3
"""
Format the test device with XFS and mount it
============================================

Formats config.fil.testing.dev with XFS, destroying what is on it, and mounts it
at config.fil.testing.mountpoint. A mount left by an earlier run at the
mountpoint is removed first; a device mounted anywhere else is refused, as it
likely holds data that is not meant for testing.

Retargetable: True
------------------
"""
import logging as log
from argparse import Namespace

from cijoe.core.command import Cijoe


def main(args: Namespace, cijoe: Cijoe):
    """Entry-point of the cijoe-script"""

    testing = cijoe.getconf("fil.testing")
    dev = testing["dev"]
    mountpoint = testing["mountpoint"]

    cijoe.run(f"umount {mountpoint} || true")

    err, _ = cijoe.run(f"findmnt --source {dev}")
    if not err:
        log.error(f"refusing to format dev({dev}); it is mounted")
        return 1

    for cmd in [
        f"mkfs.xfs -f {dev}",
        f"mkdir -p {mountpoint}",
        f"mount {dev} {mountpoint}",
    ]:
        err, _ = cijoe.run(cmd)
        if err:
            return err

    return 0
