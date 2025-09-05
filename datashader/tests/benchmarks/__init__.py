import itertools
import pytest
import numpy as np

pytestmark = pytest.mark.benchmark

array_modules = [np]
sizes = [256, 512, 1024, 2048, 4096, 8192]

try:
    import dask
    import dask.array as da

    dask.config.set(scheduler="single-threaded")
    array_modules.append(da)
except ImportError:
    da = None

try:
    import cupy

    array_modules.append(cupy)
except ImportError:
    cupy = None

array_params = []
for s, m in itertools.product(sizes, array_modules):
    if m is cupy:
        array_params.append(pytest.param((s, m), marks=pytest.mark.gpu))
    else:
        array_params.append((s, m))


def _make_id(param):
    size, array_module = param
    module_name = (
        array_module.__name__.split(".")[0]
        if hasattr(array_module, "__name__")
        else str(array_module)
    )
    return f"{size}x{size}-{module_name}"


array_fixtures = pytest.fixture(params=array_params, ids=_make_id)
