#!/usr/bin/env python3
"""
Install the CUDA toolkit
========================

Installs the parts of the CUDA toolkit that fil and xNVMe compile and link
against, including cuFile, from NVIDIA's apt repository. The version and
the distribution of the apt repository are taken from config.cuda.

When no NVIDIA driver is installed, there is no libcuda.so.1 to load at
runtime, and the toolkit stub is made loadable under that name instead. Any
CUDA call then fails, while everything not touching the GPU works.

Run as root on the target.

Retargetable: True
------------------
"""
import logging as log
from argparse import Namespace

from cijoe.core.command import Cijoe


def main(args: Namespace, cijoe: Cijoe):
    """Entry-point of the cijoe-script"""

    # Read as a section; getconf("cuda.version") would take CUDA_VERSION from the
    # environment, which CUDA images set to a version string of their own
    cuda = cijoe.getconf("cuda")
    version = cuda["version"]
    apt_distro = cuda["apt_distro"]
    path = cuda["path"]
    libdir = f"{path}/targets/x86_64-linux/lib"
    keyring = (
        "https://developer.download.nvidia.com/compute/cuda/repos/"
        f"{apt_distro}/x86_64/cuda-keyring_1.1-1_all.deb"
    )
    pkgs = [
        f"cuda-cudart-dev-{version}",
        f"cuda-driver-dev-{version}",
        f"cuda-nvcc-{version}",
        f"libcufile-dev-{version}",
    ]

    commands = [
        f"wget -qO /tmp/cuda-keyring.deb {keyring}",
        "dpkg -i /tmp/cuda-keyring.deb",
        "apt-get -qy update",
        f"DEBIAN_FRONTEND=noninteractive apt-get -qy install {' '.join(pkgs)}",
        f"ln -sfn /usr/local/cuda-{version.replace('-', '.')} {path}",
        f"echo {libdir} | tee /etc/ld.so.conf.d/cuda.conf",
        "ldconfig",
        f"{path}/bin/nvcc --version",
    ]
    for cmd in commands:
        err, _ = cijoe.run(cmd)
        if err:
            return err

    err, _ = cijoe.run("ldconfig -p | grep 'libcuda.so.1 '")
    if not err:
        log.info("libcuda.so.1 is provided by a driver; not using the stub")
        return 0

    commands = [
        "mkdir -p /usr/local/cuda-stub",
        f"ln -sfn {libdir}/stubs/libcuda.so /usr/local/cuda-stub/libcuda.so.1",
        "echo /usr/local/cuda-stub | tee /etc/ld.so.conf.d/cuda-stub.conf",
        "ldconfig",
    ]
    for cmd in commands:
        err, _ = cijoe.run(cmd)
        if err:
            return err

    return 0
