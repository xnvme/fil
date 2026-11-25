"""
Read the dataset prepared by the verify workflow with the posix backend, direct and
buffered, which reads the files through the mount and, without a copy to the GPU,
into host memory. It opens no device and lists the dataset through the mount. The
cufile backend reads into GPU memory, and the aisio and opends backends read the
device through homi with the index of xal-server, so none of those runs here.
"""

import pytest

BACKENDS = [("posix", ""), ("posix", "--buffered")]
BACKEND_IDS = ["posix", "posix-buffered"]


@pytest.fixture
def testing(cijoe):
    """config.fil.testing, with the absolute path and number of files of the dataset"""

    testing = dict(cijoe.getconf("fil.testing"))
    testing["path"] = f"{testing['mountpoint']}/{testing['data_dir']}"
    testing["verify"] = (
        f"{cijoe.getconf('fil.source')}/cijoe/auxiliary/verify_dataset.py"
    )

    err, state = cijoe.run(f"find {testing['path']} -type f | wc -l")
    assert not err, "no dataset; run the prep steps first"
    testing["n_files"] = int(state.output().strip())
    assert testing["n_files"]

    return testing


@pytest.mark.parametrize("backend,extra", BACKENDS, ids=BACKEND_IDS)
def test_filperf(cijoe, testing, backend, extra):
    """filperf reads batches beyond the size of the dataset and reports all of it"""

    err, state = cijoe.run(
        f"filperf --backend {backend} --data-dir {testing['path']} "
        f"--batch-size 8 --batches 32 --warmup 2 --summary {extra}"
    )
    assert not err
    assert f"Number of files in the dataset: {testing['n_files']}\n" in state.output()


@pytest.mark.parametrize("backend,extra", BACKENDS, ids=BACKEND_IDS)
def test_verify_content(cijoe, testing, backend, extra):
    """Every buffer from the Python binding matches a file on disk and its label"""

    venv = cijoe.getconf("fil.venv")

    err, _ = cijoe.run(
        f"{venv}/bin/python {testing['verify']} --data-dir {testing['path']} "
        f"--backend {backend} {extra}"
    )
    assert not err
