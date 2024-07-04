import sys
from contextlib import contextmanager
from importlib import reload
from importlib.util import find_spec

import pytest

__all__ = ("dask_switcher", "DASK_UNAVAILABLE", "EXPR_UNAVAILABLE", "dask_skip")

DASK_UNAVAILABLE = find_spec("dask") is None
EXPR_UNAVAILABLE = find_spec("dask_expr") is None

dask_skip = pytest.mark.skipif(DASK_UNAVAILABLE, reason="dask is not available")

@contextmanager
def dask_switcher(*, query=False, extras=None):
    """
    Context manager to switch on/off dask-expr query planning.

    Using a context manager as it is an easy way to
    change the function to a decorator.
    """
    if DASK_UNAVAILABLE:
        pytest.skip("dask is not available")
    if query and EXPR_UNAVAILABLE:
        pytest.skip("dask-expr is not available")

    import dask

    dask.config.set(**{"dataframe.query-planning": query})
    for module in ("dask.dataframe", *(extras or ())):
        if module in sys.modules:
            reload(sys.modules[module])
    yield
