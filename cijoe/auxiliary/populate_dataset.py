#!/usr/bin/env python3
"""
Populate a fil dataset

Write <mnt>/<data-dir>/class_<n>/file_<m> with varied sizes, covering sub-block,
block-aligned and block+1 lengths
"""
import argparse
import os
import random
import sys

EDGE_SIZES = [16, 511, 512, 4095, 4096, 4097, 65536, 131073]


def populate(args):
    rng = random.Random(args.seed)
    root = os.path.join(args.mnt, args.data_dir)

    for cls in range(args.classes):
        cls_dir = os.path.join(root, f"class_{cls}")
        os.makedirs(cls_dir, exist_ok=True)
        for idx in range(args.files):
            if idx < len(EDGE_SIZES):
                size = EDGE_SIZES[idx]
            else:
                size = rng.randint(16, args.max_size)
            # Prefix with the name so that no two files share content
            name = f"class_{cls}/file_{idx:04d}".encode().ljust(16, b"\0")
            data = (name + rng.randbytes(size))[:size]
            with open(os.path.join(cls_dir, f"file_{idx:04d}"), "wb") as f:
                f.write(data)

    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnt", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--files", type=int, default=32)
    parser.add_argument("--max-size", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=0)

    return populate(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
