from __future__ import annotations


from datashader import datashape
from datashader.utils import dshape_from_narwhals, dshape_from_pandas
import pytest

import datashader as ds
import numpy as np
import xarray as xr
from numpy import nan
import narwhals as nw

from datashader.tests.test_pandas import (
    _pandas,
    assert_eq_ndarray,
    c,
    c_logx,
    c_logy,
    c_logxy,
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

def test_categorical_min_n_row_index(df):
    solution = np.array([[[[0, 4, -1], [1, -1, -1], [2, -1, -1], [3, -1, -1]],
                          [[12, -1, -1], [13, -1, -1], [10, 14, -1], [11, -1, -1]]],
                         [[[8, -1, -1], [5, 9, -1], [6, -1, -1], [7, -1, -1]],
                          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [15, 19, -1]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds._min_n_row_index(n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds._min_row_index())).data)


def test_categorical_max_n_row_index(df):
    solution = np.array([[[[4, 0, -1], [1, -1, -1], [2, -1, -1], [3, -1, -1]],
                          [[12, -1, -1], [13, -1, -1], [14, 10, -1], [11, -1, -1]]],
                         [[[8, -1, -1], [9, 5, -1], [6, -1, -1], [7, -1, -1]],
                          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [19, 15, -1]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds._max_n_row_index(n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds._max_row_index())).data)


def test_categorical_first(df):
    solution = np.array([[[0, -1, nan, -3],
                          [12, -13, 10, -11]],
                         [[8, -5, 6, -7],
                          [16, -17, 18, -15]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.first("plusminus")))
        assert_eq_ndarray(agg.data, solution)


def test_categorical_last(df):
    solution = np.array([[[4, -1, nan, -3],
                          [12, -13, 14, -11]],
                         [[8, -9, 6, -7],
                          [16, -17, 18, -19]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.last("plusminus")))
        assert_eq_ndarray(agg.data, solution)


def test_categorical_first_n(df):
    solution = np.array([[[[0, 4, nan], [-1, nan, nan], [nan, nan, nan], [-3, nan, nan]],
                          [[12, nan, nan], [-13, nan, nan], [10, 14, nan], [-11, nan, nan]]],
                         [[[8, nan, nan], [-5, -9, nan], [6, nan, nan], [-7, nan, nan]],
                          [[16, nan, nan], [-17, nan, nan], [18, nan, nan], [-15, -19, nan]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.first_n("plusminus", n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.first("plusminus"))).data)


def test_categorical_last_n(df):
    solution = np.array([[[[4, 0, nan], [-1, nan, nan], [nan, nan, nan], [-3, nan, nan]],
                          [[12, nan, nan], [-13, nan, nan], [14, 10, nan], [-11, nan, nan]]],
                         [[[8, nan, nan], [-9, -5, nan], [6, nan, nan], [-7, nan, nan]],
                          [[16, nan, nan], [-17, nan, nan], [18, nan, nan], [-19, -15, nan]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.last_n("plusminus", n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.last("plusminus"))).data)


def test_where_max(df):
    out = xr.DataArray([[16, 6], [11, 1]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('i32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('i64'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('f32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('f64'), 'reverse')), out)

    # Using row index.
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('i32'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('i64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('f64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('f32'))), out)


def test_where_min(df):
    out = xr.DataArray([[20, 10], [15, 5]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('i32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('i64'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('f32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('f64'), 'reverse')), out)

    # Using row index.
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('i32'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('i64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('f64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('f32'))), out)


def test_where_max_n(df):
    sol_rowindex = np.array([[[ 4,  0,  1,  3, -1, -1],
                              [14, 12, 10, 11, 13, -1]],
                             [[ 8,  6,  5,  7,  9, -1],
                              [18, 16, 15, 17, 19, -1]]])
    sol_reverse = np.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 7):
        # Using row index.
        agg = c.points(df, 'x', 'y', ds.where(ds.max_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(df, 'x', 'y', ds.where(ds.max('plusminus'))).data)

        # Using another column
        agg = c.points(df, 'x', 'y', ds.where(ds.max_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(df, 'x', 'y', ds.where(ds.max('plusminus'),
                                                               'reverse')).data)


def test_where_min_n(df):
    sol_rowindex = np.array([[[3,  1,  0,  4, -1, -1],
                              [13, 11, 10, 12, 14, -1]],
                             [[ 9,  7,  5,  6,  8, -1],
                              [19, 17, 15, 16, 18, -1]]])
    sol_reverse = np.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 7):
        # Using row index.
        agg = c.points(df, 'x', 'y', ds.where(ds.min_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(df, 'x', 'y', ds.where(ds.min('plusminus'))).data)

        # Using another column
        agg = c.points(df, 'x', 'y', ds.where(ds.min_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(df, 'x', 'y', ds.where(ds.min('plusminus'),
                                                               'reverse')).data)


def test_where_first(df):
    # Note reductions like ds.where(ds.first('i32'), 'reverse') are supported,
    # but the same results can be achieved using the simpler ds.first('reverse')
    out = xr.DataArray([[20, 10], [15, 5]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('i32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('i64'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('f32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('f64'), 'reverse')), out)

    # Using row index.
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('i32'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('i64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('f64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('f32'))), out)


def test_where_last(df):
    # Note reductions like ds.where(ds.last('i32'), 'reverse') are supported,
    # but the same results can be achieved using the simpler ds.last('reverse')
    out = xr.DataArray([[16, 6], [11, 1]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('i32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('i64'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('f32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('f64'), 'reverse')), out)

    # Using row index.
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('i32'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('i64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('f64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('f32'))), out)

def test_where_first_n(df):
    sol_rowindex = np.array([[[ 0,  1,  3,  4, -1, -1],
                              [10, 11, 12, 13, 14, -1]],
                             [[ 5,  6,  7,  8,  9, -1],
                              [15, 16, 17, 18, 19, -1]]])
    sol_reverse = np.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 7):
        # Using row index.
        agg = c.points(df, 'x', 'y', ds.where(ds.first_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(df, 'x', 'y', ds.where(ds.first('plusminus'))).data)

        # Using another column
        agg = c.points(df, 'x', 'y', ds.where(ds.first_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(df, 'x', 'y', ds.where(ds.first('plusminus'),
                                                               'reverse')).data)


def test_where_last_n(df):
    sol_rowindex = np.array([[[ 4,  3,  1,  0, -1, -1],
                              [14, 13, 12, 11, 10, -1]],
                             [[ 9,  8,  7,  6,  5, -1],
                              [19, 18, 17, 16, 15, -1]]])
    sol_reverse = np.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 7):
        # Using row index.
        agg = c.points(df, 'x', 'y', ds.where(ds.last_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(df, 'x', 'y', ds.where(ds.last('plusminus'))).data)

        # Using another column
        agg = c.points(df, 'x', 'y', ds.where(ds.last_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(df, 'x', 'y', ds.where(ds.last('plusminus'),
                                                               'reverse')).data)

def test_summary_by(df):

    # summary(by)
    agg_summary = c.points(df, 'x', 'y', ds.summary(by=ds.by("cat")))
    agg_by = c.points(df, 'x', 'y', ds.by("cat"))
    assert_eq_xr(agg_summary["by"], agg_by)

    # summary(by, other_reduction)
    agg_summary = c.points(df, 'x', 'y', ds.summary(by=ds.by("cat"), max=ds.max("plusminus")))
    agg_max = c.points(df, 'x', 'y', ds.max("plusminus"))
    assert_eq_xr(agg_summary["by"], agg_by)
    assert_eq_xr(agg_summary["max"], agg_max)

    # summary(other_reduction, by)
    agg_summary = c.points(df, 'x', 'y', ds.summary(max=ds.max("plusminus"), by=ds.by("cat")))
    assert_eq_xr(agg_summary["by"], agg_by)
    assert_eq_xr(agg_summary["max"], agg_max)

    # summary(by, by)
    agg_summary = c.points(df, 'x', 'y', ds.summary(by=ds.by("cat"),
                                                     by_any=ds.by("cat", ds.any())))
    agg_by_any = c.points(df, 'x', 'y', ds.by("cat", ds.any()))
    assert_eq_xr(agg_summary["by"], agg_by)
    assert_eq_xr(agg_summary["by_any"], agg_by_any)

    # summary(by("cat1"), by("cat2"))
    agg_summary = c.points(df, 'x', 'y', ds.summary(by=ds.by("cat"), by2=ds.by("cat2")))
    agg_by2 = c.points(df, 'x', 'y', ds.by("cat2"))
    assert_eq_xr(agg_summary["by"], agg_by)
    assert_eq_xr(agg_summary["by2"], agg_by2)


def test_summary_where_n(df):
    sol_min_n_rowindex = np.array([[[3,  1,  0,  4, -1],
                                    [13, 11, 10, 12, 14]],
                                   [[ 9,  7,  5,  6,  8],
                                    [19, 17, 15, 16, 18]]])
    sol_max_n_rowindex = np.array([[[ 4,  0,  1,  3, -1],
                                    [14, 12, 10, 11, 13]],
                                   [[ 8,  6,  5,  7,  9],
                                    [18, 16, 15, 17, 19]]])
    sol_max_n_reverse = np.where(np.logical_or(sol_max_n_rowindex < 0, sol_max_n_rowindex == 6),
                                 np.nan, 20 - sol_max_n_rowindex)

    agg = c.points(df, 'x', 'y', ds.summary(
        count=ds.count(),
        min_n=ds.where(ds.min_n('plusminus', 5)),
        max_n=ds.where(ds.max_n('plusminus', 5), 'reverse'),
    ))
    assert_eq_ndarray(agg.coords['n'], np.arange(5))

    assert agg['count'].dims == ('y', 'x')
    assert agg['min_n'].dims == ('y', 'x', 'n')
    assert agg['max_n'].dims == ('y', 'x', 'n')

    assert agg['count'].dtype == np.dtype('uint32')
    assert agg['min_n'].dtype == np.dtype('int64')
    assert agg['max_n'].dtype == np.dtype('float64')

    assert_eq_ndarray(agg['count'].data, [[5, 5], [5, 5]])
    assert_eq_ndarray(agg['min_n'].data, sol_min_n_rowindex)
    assert_eq_ndarray(agg['max_n'].data, sol_max_n_reverse)


def test_mean(df):
    out = xr.DataArray(
        values(_pandas().i32).reshape((2, 2, 5)).mean(axis=2, dtype='f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.mean('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.mean('i64')), out)
    out = xr.DataArray(
        np.nanmean(values(_pandas().f64).reshape((2, 2, 5)), axis=2).T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.mean('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.mean('f64')), out)


def test_var(df):
    out = xr.DataArray(
        values(_pandas().i32).reshape((2, 2, 5)).var(axis=2, dtype='f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.var('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.var('i64')), out)
    out = xr.DataArray(
        np.nanvar(values(_pandas().f64).reshape((2, 2, 5)), axis=2).T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.var('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.var('f64')), out)


def test_std(df):
    out = xr.DataArray(
        values(_pandas().i32).reshape((2, 2, 5)).std(axis=2, dtype='f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.std('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.std('i64')), out)
    out = xr.DataArray(
        np.nanstd(values(_pandas().f64).reshape((2, 2, 5)), axis=2).T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.std('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.std('f64')), out)

def test_count_cat(df):
    sol = np.array([[[5, 0, 0, 0],
                     [0, 0, 5, 0]],
                    [[0, 5, 0, 0],
                     [0, 0, 0, 5]]])
    out = xr.DataArray(
        sol, coords=(coords + [['a', 'b', 'c', 'd']]), dims=(dims + ['cat'])
    )
    agg = c.points(df, 'x', 'y', ds.count_cat('cat'))
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 1), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)

    # categorizing by (cat_int-10)%4 ought to give the same result
    out = xr.DataArray(
        sol, coords=(coords + [range(4)]), dims=(dims + ['cat_int'])
    )
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.count()))
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 1), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)

    # easier to write these tests in here, since we expect the same result with only slight tweaks

    # add an extra category (this will count nans and out of bounds)
    sol = np.append(sol, [[[0], [0]],[[0], [0]]], axis=2)

    # categorizing by binning the integer arange columns using [0,20] into 4 bins. Same result as
    # for count_cat
    for col in 'i32', 'i64':
        out = xr.DataArray(
            sol, coords=(coords + [range(5)]), dims=(dims + [col])
        )
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.count()))
        assert_eq_xr(agg, out)
        assert_eq_ndarray(agg.x_range, (0, 1), close=True)
        assert_eq_ndarray(agg.y_range, (0, 1), close=True)

    # as above, but for the float arange columns. Element 2 has a nan, so the first bin is one
    # short, and the nan bin is +1
    sol[0, 0, 0] = 4
    sol[0, 0, 4] = 1

    for col in 'f32', 'f64':
        out = xr.DataArray(
            sol, coords=(coords + [range(5)]), dims=(dims + [col])
        )
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.count()))
        assert_eq_xr(agg, out)
        assert_eq_ndarray(agg.x_range, (0, 1), close=True)
        assert_eq_ndarray(agg.y_range, (0, 1), close=True)


def test_categorical_sum(df):
    sol = np.array([[[ 10, nan, nan, nan],
                     [nan, nan,  60, nan]],
                    [[nan,  35, nan, nan],
                     [nan, nan, nan,  85]]])
    out = xr.DataArray(
        sol, coords=(coords + [['a', 'b', 'c', 'd']]), dims=(dims + ['cat'])
    )
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.sum('i32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.sum('i64')))
    assert_eq_xr(agg, out)

    out = xr.DataArray(
        sol, coords=(coords + [range(4)]), dims=(dims + ['cat_int'])
    )
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.sum('i32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.sum('i64')))
    assert_eq_xr(agg, out)

    sol = np.array([[[8.0,  nan,  nan,  nan],
                     [nan,  nan, 60.0,  nan]],
                    [[nan, 35.0,  nan,  nan],
                     [nan,  nan,  nan, 85.0]]])
    out = xr.DataArray(
        sol, coords=(coords + [['a', 'b', 'c', 'd']]), dims=(dims + ['cat'])
    )
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.sum('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.sum('f64')))
    assert_eq_xr(agg, out)


def test_categorical_sum_binning(df):
    sol = np.array([[[8.0,  nan,  nan,  nan],
                     [nan,  nan, 60.0,  nan]],
                    [[nan, 35.0,  nan,  nan],
                     [nan,  nan,  nan, 85.0]]])

    # add an extra category (this will count nans and out of bounds)
    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(
            sol, coords=(coords + [range(5)]), dims=(dims + [col])
        )
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.sum(col)))
        assert_eq_xr(agg, out)
        assert_eq_ndarray(agg.x_range, (0, 1), close=True)
        assert_eq_ndarray(agg.y_range, (0, 1), close=True)


def test_categorical_mean(df):
    sol = np.array([[[  2, nan, nan, nan],
                     [nan, nan,  12, nan]],
                    [[nan,   7, nan, nan],
                     [nan, nan, nan,  17]]])
    out = xr.DataArray(
        sol,
        coords=(coords + [['a', 'b', 'c', 'd']]),
        dims=(dims + ['cat']))

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.mean('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.mean('f64')))
    assert_eq_xr(agg, out)

    out = xr.DataArray(
        sol, coords=(coords + [range(4)]), dims=(dims + ['cat_int'])
    )
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.mean('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.mean('f64')))
    assert_eq_xr(agg, out)


def test_categorical_mean_binning(df):
    sol = np.array([[[  2, nan, nan, nan],
                     [nan, nan,  12, nan]],
                    [[nan,   7, nan, nan],
                     [nan, nan, nan,  17]]])

    # add an extra category (this will count nans and out of bounds)
    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(
            sol, coords=(coords + [range(5)]), dims=(dims + [col])
        )
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.mean(col)))
        assert_eq_xr(agg, out)


def test_categorical_var(df):
    sol = np.array([[[ 2.5,  nan,  nan,  nan],
                     [ nan,  nan,   2.,  nan]],
                    [[ nan,   2.,  nan,  nan],
                     [ nan,  nan,  nan,   2.]]])
    out = xr.DataArray(
        sol,
        coords=(coords + [['a', 'b', 'c', 'd']]),
        dims=(dims + ['cat']))

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.var('f32')))
    assert_eq_xr(agg, out, True)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.var('f64')))
    assert_eq_xr(agg, out, True)

    out = xr.DataArray(
        sol, coords=(coords + [range(4)]), dims=(dims + ['cat_int'])
    )
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.var('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.var('f64')))
    assert_eq_xr(agg, out)

    # add an extra category (this will count nans and out of bounds)
    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(
            sol, coords=(coords + [range(5)]), dims=(dims + [col])
        )
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.var(col)))
        assert_eq_xr(agg, out)


def test_categorical_std(df):
    sol = np.sqrt(np.array([
        [[ 2.5,  nan,  nan,  nan],
         [ nan,  nan,   2.,  nan]],
        [[ nan,   2.,  nan,  nan],
         [ nan,  nan,  nan,   2.]]])
    )
    out = xr.DataArray(
        sol,
        coords=(coords + [['a', 'b', 'c', 'd']]),
        dims=(dims + ['cat']))

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.std('f32')))
    assert_eq_xr(agg, out, True)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.std('f64')))
    assert_eq_xr(agg, out, True)

    out = xr.DataArray(
        sol, coords=(coords + [range(4)]), dims=(dims + ['cat_int'])
    )
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.std('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.std('f64')))
    assert_eq_xr(agg, out)

    # add an extra category (this will count nans and out of bounds)
    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(
            sol, coords=(coords + [range(5)]), dims=(dims + [col])
        )
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.std(col)))
        assert_eq_xr(agg, out)

def test_multiple_aggregates(df):
    agg = c.points(df, 'x', 'y',
                   ds.summary(f64_std=ds.std('f64'),
                              f64_mean=ds.mean('f64'),
                              i32_sum=ds.sum('i32'),
                              i32_count=ds.count('i32')))

    def f(x):
        return xr.DataArray(x, coords=coords, dims=dims)
    assert_eq_xr(agg.f64_std, f(np.nanstd(values(_pandas().f64).reshape((2, 2, 5)), axis=2).T))
    assert_eq_xr(agg.f64_mean, f(np.nanmean(values(_pandas().f64).reshape((2, 2, 5)), axis=2).T))
    assert_eq_xr(agg.i32_sum, f(values(_pandas().i32).reshape((2, 2, 5)).sum(axis=2, dtype='f8').T))
    assert_eq_xr(agg.i32_count, f(np.array([[5, 5], [5, 5]], dtype='i4')))

    assert_eq_ndarray(agg.x_range, (0, 1), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)

def test_log_axis_points(df):
    axis = ds.core.LogAxis()
    logcoords = axis.compute_index(axis.compute_scale_and_translate((1, 10), 2), 2)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)

    sol = np.array([[5, 5], [5, 5]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, logcoords],
                       dims=['y', 'log_x'])
    assert_eq_xr(c_logx.points(df, 'log_x', 'y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, lincoords],
                       dims=['log_y', 'x'])
    assert_eq_xr(c_logy.points(df, 'x', 'log_y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, logcoords],
                       dims=['log_y', 'log_x'])
    assert_eq_xr(c_logxy.points(df, 'log_x', 'log_y', ds.count('i32')), out)

def test_log_axis_line(df):
    axis = ds.core.LogAxis()
    logcoords = axis.compute_index(axis.compute_scale_and_translate((1, 10), 2), 2)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)

    sol = np.array([[4, 5], [5, 5]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, logcoords],
                       dims=['y', 'log_x'])

    assert_eq_xr(c_logx.line(df, 'log_x', 'y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, lincoords],
                       dims=['log_y', 'x'])
    assert_eq_xr(c_logy.line(df, 'x', 'log_y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, logcoords],
                       dims=['log_y', 'log_x'])
    assert_eq_xr(c_logxy.line(df, 'log_x', 'log_y', ds.count('i32')), out)

@pytest.mark.parametrize('reduction,dtype,aa_dtype', [
    (ds.any(), bool, np.float32),
    (ds.count(), np.uint32, np.float32),
    (ds.max("f64"), np.float64, np.float64),
    (ds.min("f64"), np.float64, np.float64),
    (ds.sum("f64"), np.float64, np.float64),
])
def test_combine_dtype(df, reduction, dtype, aa_dtype):
    cvs = ds.Canvas(plot_width=10, plot_height=10)

    # Non-antialiased lines
    agg = cvs.line(df, 'x', 'y', line_width=0, agg=reduction)
    assert agg.dtype == dtype

    # Antialiased lines
    agg = cvs.line(df, 'x', 'y', line_width=1, agg=reduction)
    assert agg.dtype == aa_dtype


@pytest.mark.parametrize('canvas', [
    ds.Canvas(x_axis_type='log'),
    ds.Canvas(x_axis_type='log', x_range=(0, 1)),
    ds.Canvas(y_axis_type='log'),
    ds.Canvas(y_axis_type='log', y_range=(0, 1)),
])
def test_log_axis_not_positive(df, canvas):
    with pytest.raises(ValueError, match='Range values must be >0 for logarithmic axes'):
        canvas.line(df, 'x', 'y')

def test_categorical_where_max(df):
    sol_rowindex = xr.DataArray([[[4, 1, -1, 3], [12, 13, 14, 11]],
                                 [[8, 5, 6, 7], [16, 17, 18, 15]]],
                                coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    # Using row index
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.max('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)

    # Using another column
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.max('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)


def test_categorical_where_min(df):
    sol_rowindex = xr.DataArray([[[0, 1, -1, 3], [12, 13, 10, 11]],
                                 [[8, 9, 6, 7], [16, 17, 18, 19]]],
                                coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    # Using row index
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.min('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)

    # Using another column
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.min('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)


def test_categorical_where_first(df):
    sol_rowindex = xr.DataArray([[[0, 1, -1, 3], [12, 13, 10, 11]],
                                 [[8, 5, 6, 7], [16, 17, 18, 15]]],
                                coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    # Using row index
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.first('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)

    # Using another column
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.first('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)


def test_categorical_where_last(df):
    sol_rowindex = xr.DataArray([[[4, 1, -1, 3], [12, 13, 14, 11]],
                                 [[8, 9, 6, 7], [16, 17, 18, 19]]],
                                coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    # Using row index
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)

    # Using another column
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)


def test_categorical_where_max_n(df):
    sol_rowindex = xr.DataArray(
        [[[[4, 0, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]],
          [[12, -1, -1], [13, -1, -1], [14, 10, -1], [11, -1, -1]]],
         [[[8, -1, -1], [5, 9, -1], [6, -1, -1], [7, -1, -1]],
          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [15, 19, -1]]]],
        coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 4):
        # Using row index
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.max_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.max('plusminus')))).data)

        # Using another column
        agg = c.points(df, 'x', 'y',
                       ds.by('cat2', ds.where(ds.max_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.max('plusminus'),
                                                              'reverse'))).data)


def test_categorical_where_min_n(df):
    sol_rowindex = xr.DataArray(
        [[[[0, 4, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]],
          [[12, -1, -1], [13, -1, -1], [10, 14, -1], [11, -1, -1]]],
         [[[8, -1, -1], [9, 5, -1], [6, -1, -1], [7, -1, -1]],
          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [19, 15, -1]]]],
        coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 4):
        # Using row index
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.min_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.min('plusminus')))).data)

        # Using another column
        agg = c.points(df, 'x', 'y',
                       ds.by('cat2', ds.where(ds.min_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.min('plusminus'),
                                                              'reverse'))).data)


def test_categorical_where_first_n(df):
    sol_rowindex = xr.DataArray(
        [[[[0, 4, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]],
          [[12, -1, -1], [13, -1, -1], [10, 14, -1], [11, -1, -1]]],
         [[[8, -1, -1], [5, 9, -1], [6, -1, -1], [7, -1, -1]],
          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [15, 19, -1]]]],
        coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 4):
        # Using row index
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.first_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.first('plusminus')))).data)

        # Using another column
        agg = c.points(df, 'x', 'y',
                       ds.by('cat2', ds.where(ds.first_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.first('plusminus'),
                                                              'reverse'))).data)


def test_categorical_where_last_n(df):
    sol_rowindex = xr.DataArray(
        [[[[4, 0, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]],
          [[12, -1, -1], [13, -1, -1], [14, 10, -1], [11, -1, -1]]],
         [[[8, -1, -1], [9, 5, -1], [6, -1, -1], [7, -1, -1]],
          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [19, 15, -1]]]],
        coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 4):
        # Using row index
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.last('plusminus')))).data)

        # Using another column
        agg = c.points(df, 'x', 'y',
                       ds.by('cat2', ds.where(ds.last_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last('plusminus'),
                                                                             'reverse'))).data)

def test_min_n(df):
    solution = np.array([[[-3, -1, 0, 4, nan, nan], [-13, -11, 10, 12, 14, nan]],
                         [[-9, -7, -5, 6, 8, nan], [-19, -17, -15, 16, 18, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.min_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.min('plusminus')).data)


def test_max_n(df):
    solution = np.array([[[4, 0, -1, -3, nan, nan], [14, 12, 10, -11, -13, nan]],
                         [[8, 6, -5, -7, -9, nan], [18, 16, -15, -17, -19, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.max_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.max('plusminus')).data)

def test_one_category(df):
    # Issue #1142.
    assert len(df['onecat'].unique()) == 1
    sol = np.array([[[5], [5]], [[5], [5]]])
    out = xr.DataArray(sol, coords=coords + [['one']], dims=(dims + ['onecat']))
    agg = c.points(df, 'x', 'y', ds.by('onecat', ds.count('i32')))
    assert agg.shape == (2, 2, 1)
    assert_eq_xr(agg, out)

def test_categorical_count_binning(df):
    sol = np.array([[[5, 0, 0, 0],
                     [0, 0, 5, 0]],
                    [[0, 5, 0, 0],
                     [0, 0, 0, 5]]])

    # add an extra category (this will count nans and out of bounds)
    sol = np.append(sol, [[[0], [0]],[[0], [0]]], axis=2)

    # categorizing by binning the integer arange columns using [0,20] into 4 bins. Same result as
    # for count_cat
    for col in 'i32', 'i64':
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=(dims + [col]))
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.count()))
        assert_eq_xr(agg, out)

    # as above, but for the float arange columns. Element 2 has a nan, so the first bin is one
    # short, and the nan bin is +1
    sol[0, 0, 0] = 4
    sol[0, 0, 4] = 1

    for col in 'f32', 'f64':
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=(dims + [col]))
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.count()))
        assert_eq_xr(agg, out)

def test_categorical_max2(df):
    sol = np.array([[[  4, nan, nan, nan],
                     [nan, nan,  14, nan]],
                    [[nan,   9, nan, nan],
                     [nan, nan, nan,  19]]])
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=(dims + ['cat']))
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.max('i32')))
    assert_eq_xr(agg, out)

    # categorizing by (cat_int-10)%4 ought to give the same result
    out = xr.DataArray(sol, coords=coords + [range(4)], dims=(dims + ['cat_int']))

    agg = c.points(df, 'x', 'y',
                   ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.max('i32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y',
                   ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.max('i64')))
    assert_eq_xr(agg, out)


def test_categorical_max_binning(df):
    sol = np.array([[[  4, nan, nan, nan],
                     [nan, nan,  14, nan]],
                    [[nan,   9, nan, nan],
                     [nan, nan, nan,  19]]])

    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=(dims + [col]))
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.max(col)))
        assert_eq_xr(agg, out)

def test_where_min_row_index(df):
    out = xr.DataArray([[0, 10], [-5, -15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds._min_row_index(), 'plusminus')), out)


def test_where_max_row_index(df):
    out = xr.DataArray([[4, 14], [-9, -19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds._max_row_index(), 'plusminus')), out)


def test_where_min_n_row_index(df):
    sol = np.array([[[  0,  -1, nan,  -3,   4, nan],
                     [ 10, -11,  12, -13,  14, nan]],
                    [[ -5,   6,  -7,   8,  -9, nan],
                     [-15,  16, -17,  18, -19, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.where(ds._min_n_row_index(n=n), 'plusminus'))
        out = sol[:, :, :n]
        print(n, agg.data.tolist())
        print(' ', out.tolist())
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(df, 'x', 'y',
                                       ds.where(ds._min_row_index(), 'plusminus')).data)


def test_where_max_n_row_index(df):
    sol = np.array([[[  4,  -3, nan,  -1,   0, nan],
                     [ 14, -13,  12, -11,  10, nan]],
                    [[ -9,   8,  -7,   6,  -5, nan],
                     [-19,  18, -17,  16, -15, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.where(ds._max_n_row_index(n=n), 'plusminus'))
        out = sol[:, :, :n]
        print(n, agg.data.tolist())
        print(' ', out.tolist())
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(df, 'x', 'y',
                                       ds.where(ds._max_row_index(), 'plusminus')).data)

def test_dshape_matches_pandas(df):
    pd_df = _pandas()
    cols = [c for c in pd_df.columns if c not in ("cat", "cat2", "onecat")]
    nw_df = nw.from_native(df).select(cols)
    assert dshape_from_narwhals(nw_df) == dshape_from_pandas(pd_df[cols])

def test_dshape_unsupported_raises():
    df = pl.DataFrame({"x": [[1, 2], [3, 4]]})  # list column
    with pytest.raises(TypeError, match=r"narwhals .* not supported"):
        dshape_from_narwhals(nw.from_native(df))

def test_sanitise_unsupported_raises():
    cvs = ds.Canvas(plot_width=2, plot_height=2)
    with pytest.raises(ValueError,
                       match="source must be a pandas or dask DataFrame, "
                       "or a narwhals-supported eager dataframe"):
        cvs.points([1, 2, 3], 'x', 'y')

def test_dshape_polars_native_categorical():
    df = pl.DataFrame({"col": ["a", "b", "a", "c", "d", "c", "d"]},
                      schema={"col": pl.Categorical})
    out = dshape_from_narwhals(nw.from_native(df))
    expected = datashape.dshape("7 * { col: categorical[['a', 'b', 'c', 'd']," \
    "type=object, ordered=False] }")
    assert out == expected
