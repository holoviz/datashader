from importlib.util import find_spec

import pytest

__all__ = ("DASK_UNAVAILABLE", "dask_skip")

DASK_UNAVAILABLE = find_spec("dask") is None

dask_skip = pytest.mark.skipif(DASK_UNAVAILABLE, reason="dask is not available")
