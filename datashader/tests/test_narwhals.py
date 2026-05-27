from __future__ import annotations


import pytest

import datashader as ds
import numpy as np
import xarray as xr

from datashader.tests.test_pandas import _pandas, c, coords,assert_eq_xr, dims

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

@pytest.fixture(params=_backends)
def df(request):
    return request.param()


def test_count(df):
    out = xr.DataArray(np.array([[5, 5], [5, 5]], dtype='i4'),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.count('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.count('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.count()), out)
    out = xr.DataArray(np.array([[4, 5], [5, 5]], dtype='i4'),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.count('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.count('f64')), out)
