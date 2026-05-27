from __future__ import annotations


import pytest

from datashader.tests.test_pandas import _pandas

pl = pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")

def _polars():
    return pl.from_pandas(_pandas())


def _pyarrow():
    return pa.Table.from_pandas(_pandas())

_backends = [
    pytest.param(_polars, id="polars"),
    pytest.param(_pyarrow, id="pyarrow"),
]
