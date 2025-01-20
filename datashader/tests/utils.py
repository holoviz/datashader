import sys
from contextlib import contextmanager
from importlib import reload
from importlib.util import find_spec
from contextlib import suppress
from functools import lru_cache

import pytest
from packaging.version import Version

__all__ = ("dask_switcher", "DASK_UNAVAILABLE", "dask_skip")

DASK_UNAVAILABLE = find_spec("dask") is None

dask_skip = pytest.mark.skipif(DASK_UNAVAILABLE, reason="dask is not available")


@lru_cache
def _dask_setup():
    """
    Set-up both dask dataframes, using lru_cahce to only do it once

    """
    import dask
    from datashader.data_libraries.dask import bypixel, dask_pipeline

    classic, expr = False, False

    # Removed in Dask 2025.1, and will raise AttributeError
    if Version(dask.__version__).release < (2025, 1, 0):
        import dask.dataframe as dd

        bypixel.pipeline.register(dd.core.DataFrame)(dask_pipeline)
        classic = True
    else:
        # dask_expr import below will now fail with:
        # cannot import name '_Frame' from 'dask.dataframe.core'
        expr = True

    with suppress(ImportError):
        import dask_expr

        bypixel.pipeline.register(dask_expr.DataFrame)(dask_pipeline)
        expr = True

    return classic, expr


@contextmanager
def dask_switcher(*, query=False, extras=None):
    """
    Context manager to switch on/off dask-expr query planning.

    Using a context manager as it is an easy way to
    change the function to a decorator.
    """
    if DASK_UNAVAILABLE:
        pytest.skip("dask is not available")

    classic, expr = _dask_setup()

    if not query and not classic:
        pytest.skip("Classic DataFrame no longer supported by dask")
    if query and not expr:
        pytest.skip("dask-expr is not available")

    import dask

    dask.config.set(**{"dataframe.query-planning": query})
    for module in ("dask.dataframe", *(extras or ())):
        if module in sys.modules:
            reload(sys.modules[module])
    yield
