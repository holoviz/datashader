from __future__ import annotations


import pytest

import datashader as ds
import numpy as np
import xarray as xr
from numpy import nan
from datashader.tests.test_pandas import (
    _pandas,
    assert_eq_ndarray,
    c,
    coords,
    assert_eq_xr,
    dims,
    values
    )

pl = pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")

def _polars():
    df_pd = _pandas()
    # Polars Categoricals share a global pool.
    # Closest pandas-like (per-column category list) behavior is Polars Enum.
    # https://docs.pola.rs/user-guide/expressions/categorical-data-and-enums/

    schema_overrides={c: pl.Enum(df_pd[c].cat.categories) for c in ["cat", "cat2", "onecat"]}
    return pl.from_pandas(df_pd, nan_to_null=False, schema_overrides=schema_overrides)


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

def test_any(df):
    out = xr.DataArray(np.array([[True, True], [True, True]]),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.any('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.any('f64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.any()), out)
    out = xr.DataArray(np.array([[True, True], [True, False]]),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.any('empty_bin')), out)


def test_sum(df):
    out = xr.DataArray(
        values(_pandas().i32).reshape((2, 2, 5)).sum(axis=2, dtype='f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.sum('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.sum('i64')), out)

    out = xr.DataArray(
        np.nansum(values(_pandas().f64).reshape((2, 2, 5)), axis=2).T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.sum('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.sum('f64')), out)


def test_first(df):
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.first('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.first('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.first('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.first('f64')), out)


def test_last(df):
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.last('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.last('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.last('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.last('f64')), out)


def test_min(df):
    out = xr.DataArray(
        values(_pandas().i64).reshape((2, 2, 5)).min(axis=2).astype('f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.min('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.min('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.min('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.min('f64')), out)


def test_max(df):
    out = xr.DataArray(
        values(_pandas().i64).reshape((2, 2, 5)).max(axis=2).astype('f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.max('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.max('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.max('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.max('f64')), out)


def test_min_row_index(df):
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds._min_row_index()), out)


def test_max_row_index(df):
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds._max_row_index()), out)

def test_first_n(df):
    solution = np.array([[[0, -1, -3, 4, nan, nan], [10, -11, 12, -13, 14, nan]],
                         [[-5, 6, -7, 8, -9, nan], [-15, 16, -17, 18, -19, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.first_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(df, 'x', 'y', ds.first('plusminus')).data)


def test_last_n(df):
    solution = np.array([[[4, -3, -1, 0, nan, nan], [14, -13, 12, -11, 10, nan]],
                         [[-9, 8, -7, 6, -5, nan], [-19, 18, -17, 16, -15, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.last_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.last('plusminus')).data)


def test_categorical_count(df):
    sol = np.array([[[2, 1, 1, 1], [1, 1, 2, 1]], [[1, 2, 1, 1], [1, 1, 1, 2]]], dtype=np.uint32)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2')).data, sol)

    # ds.summary(name=ds.by("cat2")) should give same result as ds.by("cat2"). Issue 1252
    dataset = c.points(df, 'x', 'y', ds.summary(name=ds.by('cat2')))
    assert_eq_ndarray(dataset["name"].data, sol)


def test_categorical_min(df):
    sol_int = np.array([[[0, 1, 2, 3], [12, 13, 10, 11]], [[8, 5, 6, 7], [16, 17, 18, 15]]],
                       dtype=np.float64)
    sol_float = np.array([[[0, 1, nan, 3], [12, 13, 10, 11]], [[8, 5, 6, 7], [16, 17, 18, 15]]])
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.min('i32'))).data, sol_int)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.min('i64'))).data, sol_int)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.min('f32'))).data, sol_float)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.min('f64'))).data, sol_float)


def test_categorical_max(df):
    sol_int = np.array([[[4, 1, 2, 3], [12, 13, 14, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]],
                       dtype=np.float64)
    sol_float = np.array([[[4, 1, nan, 3], [12, 13, 14, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]])
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.max('i32'))).data, sol_int)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.max('i64'))).data, sol_int)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.max('f32'))).data, sol_float)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.max('f64'))).data, sol_float)


def test_categorical_min_n(df):
    solution = np.array([[[[0, 4, nan], [1, nan, nan], [nan, nan, nan], [3, nan, nan]],
                          [[12, nan, nan], [13, nan, nan], [10, 14, nan], [11, nan, nan]]],
                         [[[8, nan, nan], [5, 9, nan], [6, nan, nan], [7, nan, nan]],
                          [[16, nan, nan], [17, nan, nan], [18, nan, nan], [15, 19, nan]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.min_n('f32', n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.min('f32'))).data)


def test_categorical_max_n(df):
    solution = np.array([[[[4, 0, np.nan], [1, nan, nan], [nan, nan, nan], [3, nan, nan]],
                          [[12, nan, nan], [13, nan, nan], [14, 10, nan], [11, nan, nan]]],
                         [[[8, nan, nan], [9, 5, nan], [6, nan, nan], [7, np.nan, nan]],
                          [[16, nan, nan], [17, nan, nan], [18, nan, nan], [19, 15, nan]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.max_n('f32', n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.max('f32'))).data)

def test_categorical_min_row_index(df):
    solution = np.array([[[0, 1, 2, 3], [12, 13, 10, 11]], [[8, 5, 6, 7], [16, 17, 18, 15]]])
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds._min_row_index()))
    assert_eq_ndarray(agg.data, solution)


def test_categorical_max_row_index(df):
    solution = np.array([[[4, 1, 2, 3], [12, 13, 14, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]])
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds._max_row_index()))
    assert_eq_ndarray(agg.data, solution)
