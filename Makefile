BUILD_DIR ?= builddir

.PHONY: all config config-debug config-subtime build install clean git-setup format format-all

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
