"""
Option combinations that fil_init() must refuse, checked through filperf on the
dataset of config.fil.testing. Each is refused for one reason only: the device uri,
where given, is never opened, and filperf takes the last --data-dir it is given.
"""

import pytest

REJECTED = [
    "--backend aisio-cpu",
    "0000:01:00.0 --backend aisio-cpu --homi-id 1",
    "0000:01:00.0 --backend aisio-cpu --xal-shm /xal_dev0",
    "--backend opends",
    "--backend cufile --buffered",
    "--backend posix --register_bufs",
    "--backend cufile --copy-to-gpu",
    "--backend nope",
    "--backend posix --data-dir fil-testing",
    "/dev/nvme0n1,/dev/nvme0n1 --backend posix",
]


@pytest.mark.parametrize("options", REJECTED, ids=lambda o: o.replace(" ", ""))
def test_rejected(cijoe, options):
    """filperf exits with an error, and not by a signal, which would be exit > 128"""

    testing = cijoe.getconf("fil.testing")
    path = f"{testing['mountpoint']}/{testing['data_dir']}"

    err, _ = cijoe.run(f"filperf --data-dir {path} {options}")
    assert err, "accepted"
    assert not 128 < err < 160, f"terminated by signal {err - 128}"
