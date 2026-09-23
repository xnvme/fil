#!/usr/bin/env python3
"""
Transfer the fil source tree to the target
==========================================

Packs the fil checkout that this cijoe environment is part of, as the files git
tracks or would track, uncommitted changes included, transfers it to the target
and unpacks it at config.fil.source, replacing what was there. This way the
target builds what is checked out, not what was last pushed.

Retargetable: True
------------------
"""
import tempfile
from argparse import Namespace
from pathlib import Path

from cijoe.core.command import Cijoe

REPOSITORY = Path(__file__).resolve().parents[2]
ARCHIVE = "/tmp/fil-source.tar.gz"


def main(args: Namespace, cijoe: Cijoe):
    """Entry-point of the cijoe-script"""

    source = cijoe.getconf("fil.source")

    with tempfile.TemporaryDirectory() as workdir:
        archive = Path(workdir) / "fil-source.tar.gz"

        err, _ = cijoe.run_local(
            "git ls-files -z --cached --others --exclude-standard "
            f"| tar --null -T - -czf {archive}",
            cwd=REPOSITORY,
        )
        if err:
            return err

        if not cijoe.put(str(archive), ARCHIVE):
            return 1

    for cmd in [
        f"rm -rf {source}",
        f"mkdir -p {source}",
        f"tar -xzf {ARCHIVE} -C {source}",
        f"rm {ARCHIVE}",
    ]:
        err, _ = cijoe.run(cmd)
        if err:
            return err

    return 0
