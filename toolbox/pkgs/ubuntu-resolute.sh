#!/usr/bin/env bash

# SPDX-FileCopyrightText: Samsung Electronics Co., Ltd
#
# SPDX-License-Identifier: BSD-3-Clause

# Install the system packages needed to build fil, and xNVMe and xal from source
#
# CUDA is not installed here; it comes from the NVIDIA repository and its version
# is chosen by whoever sets up the system. bpftool, which xal needs, is provided by
# the linux-tools package of the running kernel. Must run as root.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export DEBIAN_PRIORITY=critical

apt-get -qy update
apt-get -qy install \
	build-essential \
	clang \
	e2fsprogs \
	git \
	libaio-dev \
	libbpf-dev \
	libelf-dev \
	libnuma-dev \
	liburing-dev \
	linux-tools-common \
	"linux-tools-$(uname -r)" \
	llvm \
	meson \
	ninja-build \
	pkg-config \
	python3-dev \
	python3-pyelftools \
	python3-venv \
	uuid-dev \
	wget \
	xfsprogs \
	xfslibs-dev \
	zlib1g-dev
