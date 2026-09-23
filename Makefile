BUILD_DIR ?= builddir
CIJOE_OUTPUT ?= cijoe-output
CIJOE_TARGET ?= configs/qemu_guest.toml

.PHONY: all config config-debug config-subtime build install clean git-setup format format-all \
	cijoe-install cijoe-guest-start cijoe-verify

all: config build install

config:
	meson setup $(BUILD_DIR)

config-debug:
	meson setup $(BUILD_DIR) --buildtype=debug

config-subtime:
	meson setup $(BUILD_DIR) -Dsubtime=true

build:
	meson compile -C $(BUILD_DIR)

install:
	meson install -C $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

# Install the pre-commit git hooks
git-setup:
	./toolbox/pre-commit-check.sh
	pre-commit install

# Run code format/lint on staged changes
format:
	pre-commit run

# Run code format/lint on all files
format-all:
	pre-commit run --all-files

# Install cijoe and the dependencies of the fil cijoe environment via pipx
cijoe-install:
	pipx install cijoe/ --include-deps --force

# Start a QEMU guest with an emulated NVMe device, the default CIJOE_TARGET; needs KVM
cijoe-guest-start:
	cd cijoe && cijoe workflows/guest_start.yaml \
		--config configs/qemu_guest.toml \
		--output $(CIJOE_OUTPUT)-guest-start \
		--monitor

# Build, install and test fil and its dependencies on the target given by
# CIJOE_TARGET, a cijoe config with a transport to it as root
cijoe-verify:
	cd cijoe && cijoe workflows/verify.yaml \
		--config $(CIJOE_TARGET) \
		--config configs/fil.toml \
		--output $(CIJOE_OUTPUT) \
		--monitor
