"""
Read the dataset prepared by prep_{xfs,dataset} with the posix backend, direct and
buffered, which reads the files through the mount and, without a copy to the GPU,
into host memory. The cufile backend reads into GPU memory, and the aisio backends
need the device on a user-space driver, so neither runs here.
"""

import pytest

BACKENDS = [("posix", ""), ("posix", "--buffered")]
BACKEND_IDS = ["posix", "posix-buffered"]


@pytest.fixture
def testing(cijoe):
    """config.fil.testing, with the number of files in the dataset"""

    testing = dict(cijoe.getconf("fil.testing"))
    testing["dataset"] = f"{cijoe.getconf('fil.source')}/cijoe/auxiliary/fil_dataset.py"

    err, state = cijoe.run(
        f"find {testing['mountpoint']}/{testing['data_dir']} -type f | wc -l"
    )
    assert not err, "no dataset; run the prep steps first"
    testing["n_files"] = int(state.output().strip())
    assert testing["n_files"]

    return testing


@pytest.mark.parametrize("backend,extra", BACKENDS, ids=BACKEND_IDS)
def test_filperf(cijoe, testing, backend, extra):
    """filperf reads batches beyond the size of the dataset and reports all of it"""

    err, state = cijoe.run(
        f"filperf {testing['dev']} --backend {backend} "
        f"--mnt {testing['mountpoint']} --data-dir {testing['data_dir']} "
        f"--batch-size 8 --batches 32 --warmup 2 --summary {extra}"
    )
    assert not err
    assert f"Number of files in the dataset: {testing['n_files']}\n" in state.output()


@pytest.mark.parametrize("backend,extra", BACKENDS, ids=BACKEND_IDS)
def test_verify_content(cijoe, testing, backend, extra):
    """Every buffer from the Python binding matches a file on disk and its label"""

    venv = cijoe.getconf("fil.venv")

    err, _ = cijoe.run(
        f"{venv}/bin/python {testing['dataset']} verify --dev {testing['dev']} "
        f"--mnt {testing['mountpoint']} --data-dir {testing['data_dir']} "
        f"--backend {backend} {extra}"
    )
    assert not err
