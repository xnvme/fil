#!/usr/bin/env python3
"""
Verify a fil dataset

Read the whole dataset through the fil Python binding and check that every
returned buffer matches a file on disk byte for byte, carries the label of its
class directory, and that every file is returned
"""
import argparse
import hashlib
import math
import os
import sys


def scan(root):
    """Map sha256 of each file to (label, path), labels follow sorted class names"""
    expected = {}
    for label, cls in enumerate(sorted(os.listdir(root))):
        cls_dir = os.path.join(root, cls)
        for name in sorted(os.listdir(cls_dir)):
            path = os.path.join(cls_dir, name)
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            if digest in expected:
                raise RuntimeError(f"{path}: same content as {expected[digest][1]}")
            expected[digest] = (label, path)

    return expected


def verify(args):
    import fil

    expected = scan(os.path.join(args.mnt, args.data_dir))

    n_files = fil.init(
        args.dev,
        data_dir=args.data_dir,
        mnt=args.mnt,
        backend=args.backend,
        batch_size=args.batch_size,
        buffered=args.buffered,
    )
    if n_files != len(expected):
        print(f"FAIL: fil reports {n_files} files, expected {len(expected)}")
        return 1

    errors = 0
    seen = set()
    for _ in range(math.ceil(n_files / args.batch_size)):
        buffers, labels = fil.next()
        for buf, label in zip(buffers, labels):
            digest = hashlib.sha256(buf.tobytes()).hexdigest()
            if digest not in expected:
                print(f"FAIL: {len(buf)} byte buffer (label {label}) matches no file")
                errors += 1
                continue
            exp_label, path = expected[digest]
            if label != exp_label:
                print(f"FAIL: {path}: label {label}, expected {exp_label}")
                errors += 1
            seen.add(digest)
    fil.term()

    for digest in expected.keys() - seen:
        print(f"FAIL: {expected[digest][1]}: never returned")
        errors += 1

    print(
        f"{args.backend}{' (buffered)' if args.buffered else ''}: "
        f"{len(seen)}/{len(expected)} files verified, {errors} errors"
    )
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--mnt", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--buffered", action="store_true")

    return verify(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
