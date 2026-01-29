from importlib.util import find_spec
from packaging.version import Version

collect_ignore_glob = [
    "tiling.ipynb",
]

if find_spec("geopandas") is None or (
    find_spec("pyogrio") is None and find_spec("fiona") is None
):
    collect_ignore_glob += [
        "user_guide/13_Geopandas.ipynb",
    ]

if find_spec("spatialpandas") is None:
    collect_ignore_glob += [
        "user_guide/7_Networks.ipynb",
        "user_guide/8_Polygons.ipynb",
    ]

if find_spec("dask") is not None:
    import dask

    # Spatialpandas does not support dask-expr, which is
    # only available from this version.
    if Version(dask.__version__).release >= (2025, 1, 0):
        collect_ignore_glob += [
            "user_guide/8_Polygons.ipynb",
        ]


def pytest_runtest_makereport(item, call):
    """
    Skip tests that fail because "the kernel died before replying to kernel_info"
    this is a common error when running the example tests in CI.

    Inspired from: https://stackoverflow.com/questions/32451811

    """
    from _pytest.runner import pytest_runtest_makereport

    tr = pytest_runtest_makereport(item, call)

    if call.excinfo is not None:
        msgs = [
            "Kernel died before replying to kernel_info",
            "Kernel didn't respond in 60 seconds",
        ]
        for msg in msgs:
            if call.excinfo.type is RuntimeError and call.excinfo.value.args[0] in msg:
                tr.outcome = "skipped"
                tr.wasxfail = f"reason: {msg}"

    return tr
