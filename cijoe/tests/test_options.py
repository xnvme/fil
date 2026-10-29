"""
Option combinations that fil_init() must refuse, checked through filperf against
config.fil.testing.dev
"""

import pytest

REJECTED = [
    "--backend aisio-cpu",
    "--backend cufile --buffered",
    "--backend posix --async",
    "--backend cufile --copy-to-gpu",
    "--backend nope",
]


@pytest.mark.parametrize("options", REJECTED, ids=lambda o: o.replace(" ", ""))
def test_rejected(cijoe, options):
    """filperf exits with an error, and not by a signal, which would be exit > 128"""

    testing = cijoe.getconf("fil.testing")

    err, _ = cijoe.run(
        f"filperf {testing['dev']} --data-dir {testing['data_dir']} {options}"
    )
    assert err, "accepted"
    assert not 128 < err < 160, f"terminated by signal {err - 128}"
