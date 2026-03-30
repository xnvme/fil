BUILD_DIR ?= builddir

.PHONY: all config config-debug build install clean

all: config build install

config:
	meson setup $(BUILD_DIR)

config-debug:
	meson setup $(BUILD_DIR) --buildtype=debug

build:
	meson compile -C $(BUILD_DIR)

install:
	meson install -C $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
