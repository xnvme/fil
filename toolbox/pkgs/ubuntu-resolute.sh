#!/usr/bin/env bash

# SPDX-FileCopyrightText: Samsung Electronics Co., Ltd
#
# SPDX-License-Identifier: BSD-3-Clause

# Install the system packages needed to build and test fil, and to build xNVMe and
# xal from source
#
# CUDA is not installed here; it comes from the NVIDIA repository and its version
# is chosen by whoever sets up the system. Must run as root.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export DEBIAN_PRIORITY=critical

apt-get -qy update
apt-get -qy install \
	build-essential \
	e2fsprogs \
	git \
	meson \
	pkg-config \
	python3-dev \
	python3-venv \
	wget \
	xfslibs-dev \
	xfsprogs
