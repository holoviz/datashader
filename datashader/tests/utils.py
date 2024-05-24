from contextlib import contextmanager
from importlib import reload
from importlib.util import find_spec

import dask
import dask.dataframe as dd
import pytest

__all__ = ("dask_switcher",)


@contextmanager
def dask_switcher(*, query=False):
    """
    Context manager to switch on/off dask-expr query planning.
    """
    if query and find_spec("dask_expr") is None:
        pytest.skip("dask-expr is not available")
    dask.config.set(**{"dataframe.query-planning": query})
    reload(dd)
    yield
