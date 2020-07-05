from __future__ import division, absolute_import

import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr

from dask.context import config
from numpy import nan

import datashader as ds
import datashader.utils as du

import pytest

try:
    import spatialpandas as sp
    import spatialpandas.dask  # noqa (API import)
except ImportError:
    sp = None

from datashader.tests.test_pandas import (
    assert_eq_xr, assert_eq_ndarray, values
)

config.set(scheduler='synchronous')

if "DATASHADER_TEST_GPU" in os.environ:
    test_gpu = bool(int(os.environ["DATASHADER_TEST_GPU"]))
else:
    test_gpu = None

df_pd = pd.DataFrame({'x': np.array(([0.] * 10 + [1] * 10)),
                      'y': np.array(([0.] * 5 + [1] * 5 + [0] * 5 + [1] * 5)),
                      'log_x': np.array(([1.] * 10 + [10] * 10)),
                      'log_y': np.array(([1.] * 5 + [10] * 5 + [1] * 5 + [10] * 5)),
                      'i32': np.arange(20, dtype='i4'),
                      'i64': np.arange(20, dtype='i8'),
                      'f32': np.arange(20, dtype='f4'),
                      'f64': np.arange(20, dtype='f8'),
                      'empty_bin': np.array([0.] * 15 + [np.nan] * 5),
                      'cat': ['a']*5 + ['b']*5 + ['c']*5 + ['d']*5})
df_pd.cat = df_pd.cat.astype('category')
df_pd.at[2,'f32'] = np.nan
df_pd.at[2,'f64'] = np.nan

_ddf = dd.from_pandas(df_pd, npartitions=2)

def dask_DataFrame(*args, **kwargs):
    if kwargs.pop("geo", False):
        df = sp.GeoDataFrame(*args, **kwargs)
    else:
        df = pd.DataFrame(*args, **kwargs)
    return dd.from_pandas(df, npartitions=2)


try:
    import cudf
    import cupy
    import dask_cudf

    if test_gpu is False:
        # GPU testing disabled even though cudf/cupy are available
        raise ImportError

    ddfs = [_ddf, dask_cudf.from_dask_dataframe(_ddf)]

    def dask_cudf_DataFrame(*args, **kwargs):
        assert not kwargs.pop("geo", False)
        cdf = cudf.DataFrame.from_pandas(
            pd.DataFrame(*args, **kwargs), nan_as_null=False
        )
        return dask_cudf.from_cudf(cdf, npartitions=2)

    DataFrames = [dask_DataFrame, dask_cudf_DataFrame]
except ImportError:
    cudf = cupy = dask_cudf = None
    ddfs = [_ddf]
    DataFrames = [dask_DataFrame]
    dask_cudf_DataFrame = None


c = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))
c_logx = ds.Canvas(plot_width=2, plot_height=2, x_range=(1, 10),
                   y_range=(0, 1), x_axis_type='log')
c_logy = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1),
                   y_range=(1, 10), y_axis_type='log')
c_logxy = ds.Canvas(plot_width=2, plot_height=2, x_range=(1, 10),
                    y_range=(1, 10), x_axis_type='log', y_axis_type='log')

axis = ds.core.LinearAxis()
lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)
coords = [lincoords, lincoords]
dims = ['y', 'x']


def assert_eq(agg, b):
    assert agg.equals(b)


def floats(n):
    """Returns contiguous list of floats from initial point"""
    while True:
        yield n
        n = n + np.spacing(n)


def test_gpu_dependencies():
    if test_gpu is True and cudf is None:
        pytest.fail(
            "cudf, cupy, and/or dask_cudf not available and DATASHADER_TEST_GPU=1"
        )


@pytest.mark.parametrize('ddf', ddfs)
def test_count(ddf):
    out = xr.DataArray(np.array([[5, 5], [5, 5]], dtype='i4'),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.count('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.count('i64')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.count()), out)
    out = xr.DataArray(np.array([[4, 5], [5, 5]], dtype='i4'),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.count('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.count('f64')), out)


@pytest.mark.parametrize('ddf', ddfs)
def test_any(ddf):
    out = xr.DataArray(np.array([[True, True], [True, True]]),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.any('i64')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.any('f64')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.any()), out)
    out = xr.DataArray(np.array([[True, True], [True, False]]),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.any('empty_bin')), out)


@pytest.mark.parametrize('ddf', ddfs)
def test_sum(ddf):
    out = xr.DataArray(
        values(df_pd.i32).reshape((2, 2, 5)).sum(axis=2, dtype='f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.sum('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.sum('i64')), out)

    out = xr.DataArray(
        np.nansum(values(df_pd.f64).reshape((2, 2, 5)), axis=2).T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.sum('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.sum('f64')), out)


@pytest.mark.parametrize('ddf', ddfs)
def test_min(ddf):
    out = xr.DataArray(
        values(df_pd.i64).reshape((2, 2, 5)).min(axis=2).astype('f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.min('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.min('i64')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.min('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.min('f64')), out)


@pytest.mark.parametrize('ddf', ddfs)
def test_max(ddf):
    out = xr.DataArray(
        values(df_pd.i64).reshape((2, 2, 5)).max(axis=2).astype('f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.max('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.max('i64')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.max('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.max('f64')), out)


@pytest.mark.parametrize('ddf', ddfs)
def test_mean(ddf):
    out = xr.DataArray(
        values(df_pd.i32).reshape((2, 2, 5)).mean(axis=2, dtype='f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.mean('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.mean('i64')), out)
    out = xr.DataArray(
        np.nanmean(values(df_pd.f64).reshape((2, 2, 5)), axis=2).T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.mean('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.mean('f64')), out)


@pytest.mark.parametrize('ddf', ddfs)
def test_var(ddf):
    if dask_cudf and isinstance(ddf, dask_cudf.DataFrame):
        pytest.skip("var not supported with cudf")

    out = xr.DataArray(
        values(df_pd.i32).reshape((2, 2, 5)).var(axis=2, dtype='f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.var('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.var('i64')), out)
    out = xr.DataArray(
        np.nanvar(values(df_pd.f64).reshape((2, 2, 5)), axis=2).T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.var('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.var('f64')), out)


@pytest.mark.parametrize('ddf', ddfs)
def test_std(ddf):
    if dask_cudf and isinstance(ddf, dask_cudf.DataFrame):
        pytest.skip("std not supported with cudf")

    out = xr.DataArray(
        values(df_pd.i32).reshape((2, 2, 5)).std(axis=2, dtype='f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.std('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.std('i64')), out)
    out = xr.DataArray(
        np.nanstd(values(df_pd.f64).reshape((2, 2, 5)), axis=2).T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.std('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.std('f64')), out)


@pytest.mark.parametrize('ddf', ddfs)
def test_count_cat(ddf):
    sol = np.array([[[5, 0, 0, 0],
                     [0, 0, 5, 0]],
                    [[0, 5, 0, 0],
                     [0, 0, 0, 5]]])
    out = xr.DataArray(
        sol, coords=(coords + [['a', 'b', 'c', 'd']]), dims=(dims + ['cat'])
    )
    agg = c.points(ddf, 'x', 'y', ds.count_cat('cat'))
    assert_eq_xr(agg, out)

@pytest.mark.parametrize('ddf', ddfs)
def test_categorical_sum(ddf):
    sol = np.array([[[ 10, nan, nan, nan],
                     [nan, nan,  60, nan]],
                    [[nan,  35, nan, nan],
                     [nan, nan, nan,  85]]])
    out = xr.DataArray(
        sol, coords=(coords + [['a', 'b', 'c', 'd']]), dims=(dims + ['cat'])
    )
    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.sum('i32')))
    assert_eq_xr(agg, out)

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.sum('i64')))
    assert_eq_xr(agg, out)

    sol = np.array([[[8.0,  nan,  nan,  nan],
                     [nan,  nan, 60.0,  nan]],
                    [[nan, 35.0,  nan,  nan],
                     [nan,  nan,  nan, 85.0]]])
    out = xr.DataArray(
        sol, coords=(coords + [['a', 'b', 'c', 'd']]), dims=(dims + ['cat'])
    )
    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.sum('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.sum('f64')))
    assert_eq_xr(agg, out)

@pytest.mark.parametrize('ddf', ddfs)
def test_categorical_mean(ddf):
    sol = np.array([[[  2, nan, nan, nan],
                     [nan, nan,  12, nan]],
                    [[nan,   7, nan, nan],
                     [nan, nan, nan,  17]]])
    out = xr.DataArray(
        sol,
        coords=(coords + [['a', 'b', 'c', 'd']]),
        dims=(dims + ['cat']))

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.mean('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.mean('f64')))
    assert_eq_xr(agg, out)

@pytest.mark.parametrize('ddf', ddfs)
def test_categorical_var(ddf):
    if cudf and isinstance(ddf._meta, cudf.DataFrame):
        pytest.skip(
            "The 'var' reduction is yet supported on the GPU"
        )

    sol = np.array([[[ 2.5,  nan,  nan,  nan],
                     [ nan,  nan,   2.,  nan]],
                    [[ nan,   2.,  nan,  nan],
                     [ nan,  nan,  nan,   2.]]])
    out = xr.DataArray(
        sol,
        coords=(coords + [['a', 'b', 'c', 'd']]),
        dims=(dims + ['cat']))

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.var('f32')))
    assert_eq_xr(agg, out, True)

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.var('f64')))
    assert_eq_xr(agg, out, True)

@pytest.mark.parametrize('ddf', ddfs)
def test_categorical_std(ddf):
    if cudf and isinstance(ddf._meta, cudf.DataFrame):
        pytest.skip(
            "The 'std' reduction is yet supported on the GPU"
        )

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

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.std('f32')))
    assert_eq_xr(agg, out, True)

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.std('f64')))
    assert_eq_xr(agg, out, True)

@pytest.mark.parametrize('ddf', ddfs)
def test_multiple_aggregates(ddf):
    if dask_cudf and isinstance(ddf, dask_cudf.DataFrame):
        pytest.skip("std not supported with cudf")

    agg = c.points(ddf, 'x', 'y',
                   ds.summary(f64_std=ds.std('f64'),
                              f64_mean=ds.mean('f64'),
                              i32_sum=ds.sum('i32'),
                              i32_count=ds.count('i32')))

    f = lambda x: xr.DataArray(x, coords=coords, dims=dims)
    assert_eq_xr(agg.f64_std, f(np.nanstd(values(df_pd.f64).reshape((2, 2, 5)), axis=2).T))
    assert_eq_xr(agg.f64_mean, f(np.nanmean(values(df_pd.f64).reshape((2, 2, 5)), axis=2).T))
    assert_eq_xr(agg.i32_sum, f(values(df_pd.i32).reshape((2, 2, 5)).sum(axis=2, dtype='f8').T))
    assert_eq_xr(agg.i32_count, f(np.array([[5, 5], [5, 5]], dtype='i4')))


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_auto_range_points(DataFrame):
    n = 10
    data = np.arange(n, dtype='i4')

    ddf = DataFrame({'time': np.arange(n),
                     'x': data,
                     'y': data})

    cvs = ds.Canvas(plot_width=n, plot_height=n)
    agg = cvs.points(ddf, 'x', 'y', ds.count('time'))
    sol = np.zeros((n, n), int)
    np.fill_diagonal(sol, 1)
    assert_eq_ndarray(agg.data, sol)

    cvs = ds.Canvas(plot_width=n+1, plot_height=n+1)
    agg = cvs.points(ddf, 'x', 'y', ds.count('time'))
    sol = np.zeros((n+1, n+1), int)
    np.fill_diagonal(sol, 1)
    sol[5, 5] = 0
    assert_eq_ndarray(agg.data, sol)

    n = 4
    data = np.arange(n, dtype='i4')
    ddf = DataFrame({'time': np.arange(n),
                     'x': data,
                     'y': data})

    cvs = ds.Canvas(plot_width=2*n, plot_height=2*n)
    agg = cvs.points(ddf, 'x', 'y', ds.count('time'))
    sol = np.zeros((2*n, 2*n), int)
    np.fill_diagonal(sol, 1)
    sol[np.array([tuple(range(1, 4, 2))])] = 0
    sol[np.array([tuple(range(4, 8, 2))])] = 0
    assert_eq_ndarray(agg.data, sol)

    cvs = ds.Canvas(plot_width=2*n+1, plot_height=2*n+1)
    agg = cvs.points(ddf, 'x', 'y', ds.count('time'))
    sol = np.zeros((2*n+1, 2*n+1), int)
    sol[0, 0] = 1
    sol[3, 3] = 1
    sol[6, 6] = 1
    sol[8, 8] = 1
    assert_eq_ndarray(agg.data, sol)


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_uniform_points(DataFrame):
    n = 101
    ddf = DataFrame({'time': np.ones(2*n, dtype='i4'),
                     'x': np.concatenate((np.arange(n, dtype='f8'),
                                          np.arange(n, dtype='f8'))),
                     'y': np.concatenate(([0.] * n, [1.] * n))})

    cvs = ds.Canvas(plot_width=10, plot_height=2, y_range=(0, 1))
    agg = cvs.points(ddf, 'x', 'y', ds.count('time'))
    sol = np.array([[10] * 9 + [11], [10] * 9 + [11]], dtype='i4')
    assert_eq_ndarray(agg.data, sol)


@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('high', [9, 10, 99, 100])
@pytest.mark.parametrize('low', [0])
def test_uniform_diagonal_points(DataFrame, low, high):
    bounds = (low, high)
    x_range, y_range = bounds, bounds

    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    n = width * height
    ddf = DataFrame({'time': np.ones(n, dtype='i4'),
                     'x': np.array([np.arange(*x_range, dtype='f8')] * width).flatten(),
                     'y': np.array([np.arange(*y_range, dtype='f8')] * height).flatten()})

    cvs = ds.Canvas(plot_width=2, plot_height=2, x_range=x_range, y_range=y_range)
    agg = cvs.points(ddf, 'x', 'y', ds.count('time'))

    diagonal = agg.data.diagonal(0)
    assert sum(diagonal) == n
    assert abs(bounds[1] - bounds[0]) % 2 == abs(diagonal[1] / high - diagonal[0] / high)


@pytest.mark.parametrize('ddf', ddfs)
def test_log_axis_points(ddf):
    axis = ds.core.LogAxis()
    logcoords = axis.compute_index(axis.compute_scale_and_translate((1, 10), 2), 2)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)

    sol = np.array([[5, 5], [5, 5]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, logcoords],
                       dims=['y', 'log_x'])
    assert_eq_xr(c_logx.points(ddf, 'log_x', 'y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, lincoords],
                       dims=['log_y', 'x'])
    assert_eq_xr(c_logy.points(ddf, 'x', 'log_y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, logcoords],
                       dims=['log_y', 'log_x'])
    assert_eq_xr(c_logxy.points(ddf, 'log_x', 'log_y', ds.count('i32')), out)


@pytest.mark.skipif(not sp, reason="spatialpandas not installed")
def test_points_geometry():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0., 2.), 3), 3)

    ddf = dd.from_pandas(sp.GeoDataFrame({
        'geom': pd.array(
            [[0, 0], [0, 1, 1, 1], [0, 2, 1, 2, 2, 2]], dtype='MultiPoint[float64]'),
        'v': [1, 2, 3]
    }), npartitions=3)

    cvs = ds.Canvas(plot_width=3, plot_height=3)
    agg = cvs.points(ddf, geometry='geom', agg=ds.sum('v'))
    sol = np.array([[1, nan, nan],
                    [2, 2,   nan],
                    [3, 3,   3]], dtype='float64')
    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_line(DataFrame):
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-3., 3.), 7), 7)

    ddf = DataFrame({'x': [4, 0, -4, -3, -2, -1.9, 0, 10, 10, 0, 4],
                     'y': [0, -4, 0, 1, 2, 2.1, 4, 20, 30, 4, 0]})
    cvs = ds.Canvas(plot_width=7, plot_height=7,
                    x_range=(-3, 3), y_range=(-3, 3))
    agg = cvs.line(ddf, 'x', 'y', ds.count())
    sol = np.array([[0, 0, 1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1],
                    [0, 2, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


# # Line tests
line_manual_range_params = [
    # axis1 none constant
    (dict(data={
        'x0': [4, -4, 4],
        'x1': [0,  0, 0],
        'x2': [-4, 4, -4],
        'y0': [0,  0, 0],
        'y1': [-4, 4, 0],
        'y2': [0,  0, 0]
    }), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis1 x constant
    (dict(data={
        'y0': [0, 0,  0],
        'y1': [0, 4, -4],
        'y2': [0, 0,  0]
    }), dict(x=np.array([-4, 0, 4]), y=['y0', 'y1', 'y2'], axis=1)),

    # axis0 single
    (dict(data={
        'x': [4, 0, -4, np.nan, -4, 0, 4, np.nan, 4, 0, -4],
        'y': [0, -4, 0, np.nan, 0, 4, 0, np.nan, 0, 0, 0],
    }), dict(x='x', y='y', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [4,  0, -4],
        'x1': [-4, 0,  4],
        'x2': [4,  0, -4],
        'y0': [0, -4,  0],
        'y1': [0,  4,  0],
        'y2': [0,  0,  0]
    }), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=0)),

    # axis0 multi with string
    (dict(data={
        'x0': [-4,  0, 4],
        'y0': [0, -4,  0],
        'y1': [0,  4,  0],
        'y2': [0,  0,  0]
    }), dict(x='x0', y=['y0', 'y1', 'y2'], axis=0)),

    # axis1 RaggedArray
    (dict(data={
        'x': [[4, 0, -4], [-4, 0, 4, 4, 0, -4]],
        'y': [[0, -4, 0], [0, 4, 0, 0, 0, 0]],
    }, dtype='Ragged[int64]'), dict(x='x', y='y', axis=1)),
]
if sp:
    line_manual_range_params.append(
        # geometry
        (dict(data={
            'geom': [[4, 0, 0, -4, -4, 0],
                     [-4, 0, 0, 4, 4, 0, 4, 0, 0, 0, -4, 0]]
        }, dtype='Line[int64]'), dict(geometry='geom'))
    )
@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_kwargs,cvs_kwargs', line_manual_range_params)
def test_line_manual_range(DataFrame, df_kwargs, cvs_kwargs):
    if DataFrame is dask_cudf_DataFrame:
        dtype = df_kwargs.get('dtype', '')
        if dtype.startswith('Ragged') or dtype.startswith('Line'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-3., 3.), 7), 7)

    ddf = DataFrame(geo='geometry' in cvs_kwargs, **df_kwargs)
    cvs = ds.Canvas(plot_width=7, plot_height=7,
                    x_range=(-3, 3), y_range=(-3, 3))

    agg = cvs.line(ddf, agg=ds.count(), **cvs_kwargs)

    sol = np.array([[0, 0, 1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 0]], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


line_autorange_params = [
    # axis1 none constant
    (dict(data={
        'x0': [0,  0, 0],
        'x1': [-4, 0, 4],
        'x2': [0,  0, 0],
        'y0': [-4, 4, -4],
        'y1': [0,  0,  0],
        'y2': [4, -4,  4]
    }), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis1 y constant
    (dict(data={
        'x0': [0,  0, 0],
        'x1': [-4, 0, 4],
        'x2': [0,  0, 0],
    }), dict(x=['x0', 'x1', 'x2'], y=np.array([-4, 0, 4]), axis=1)),

    # axis0 single
    (dict(data={
        'x': [0, -4, 0, np.nan, 0, 0, 0, np.nan, 0, 4, 0],
        'y': [-4, 0, 4, np.nan, 4, 0, -4, np.nan, -4, 0, 4],
    }), dict(x='x', y='y', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [0, -4,  0],
        'x1': [0,  0,  0],
        'x2': [0,  4,  0],
        'y0': [-4, 0,  4],
        'y1': [4,  0, -4],
        'y2': [-4, 0,  4]
    }), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=0)),

    # axis0 multi with string
    (dict(data={
        'x0': [0, -4,  0],
        'x1': [0,  0,  0],
        'x2': [0,  4,  0],
        'y0': [-4, 0,  4]
    }), dict(x=['x0', 'x1', 'x2'], y='y0', axis=0)),

    # axis1 RaggedArray
    (dict(data={
        'x': [[0, -4, 0], [0, 0, 0], [0, 4, 0]],
        'y': [[-4, 0, 4], [4, 0, -4], [-4, 0, 4]],
    }, dtype='Ragged[int64]'), dict(x='x', y='y', axis=1)),
]
if sp:
    line_autorange_params.append(
        # geometry
        (dict(data={
            'geom': [[0, -4, -4, 0, 0, 4],
                     [0, 4, 0, 0, 0, -4],
                     [0, -4, 4, 0, 0, 4]]
        }, dtype='Line[int64]'), dict(geometry='geom'))
    )
@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_kwargs,cvs_kwargs', line_autorange_params)
def test_line_autorange(DataFrame, df_kwargs, cvs_kwargs):
    if DataFrame is dask_cudf_DataFrame:
        dtype = df_kwargs.get('dtype', '')
        if dtype.startswith('Ragged') or dtype.startswith('Line'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 9), 9)

    ddf = DataFrame(geo='geometry' in cvs_kwargs, **df_kwargs)

    cvs = ds.Canvas(plot_width=9, plot_height=9)

    agg = cvs.line(ddf, agg=ds.count(), **cvs_kwargs)

    sol = np.array([[0, 0, 0, 0, 3, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 1, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 3, 0, 0, 0, 0]], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_line_x_constant_autorange(DataFrame):
    # axis1 y constant
    x = np.array([-4, 0, 4])
    y = ['y0', 'y1', 'y2']
    ax = 1

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 9), 9)

    ddf = DataFrame({
        'y0': [0, 0, 0],
        'y1': [-4, 0, 4],
        'y2': [0, 0, 0],
    })

    cvs = ds.Canvas(plot_width=9, plot_height=9)

    agg = cvs.line(ddf, x, y, ds.count(), axis=ax)

    sol = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [3, 1, 1, 1, 1, 1, 1, 1, 3],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('ddf', ddfs)
def test_log_axis_line(ddf):
    axis = ds.core.LogAxis()
    logcoords = axis.compute_index(axis.compute_scale_and_translate((1, 10), 2), 2)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)

    sol = np.array([[4, 5], [5, 5]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, logcoords],
                       dims=['y', 'log_x'])

    assert_eq_xr(c_logx.line(ddf, 'log_x', 'y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, lincoords],
                       dims=['log_y', 'x'])
    assert_eq_xr(c_logy.line(ddf, 'x', 'log_y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, logcoords],
                       dims=['log_y', 'log_x'])
    assert_eq_xr(c_logxy.line(ddf, 'log_x', 'log_y', ds.count('i32')), out)


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_auto_range_line(DataFrame):
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-10., 10.), 5), 5)

    ddf = DataFrame({'x': [-10,  0, 10,   0, -10],
                     'y': [  0, 10,  0, -10,   0]})

    cvs = ds.Canvas(plot_width=5, plot_height=5)
    agg = cvs.line(ddf, 'x', 'y', ds.count())
    sol = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [2, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [
    # axis1 none constant
    (dict(data={
        'x0': [-4, np.nan],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, np.nan],
        'y1': [-4, 4],
        'y2': [0, 0]
    }, dtype='float32'), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis0 single
    (dict(data={
        'x': [-4, -2, 0, np.nan, 2, 4],
        'y': [0, -4, 0, np.nan, 4, 0],
    }), dict(x='x', y='y', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [np.nan, 2, 4],
        'y0': [0, -4, 0],
        'y1': [np.nan, 4, 0],
    }, dtype='float32'),  dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': pd.array([[-4, -2, 0], [2, 4]]),
        'y': pd.array([[0, -4, 0], [4, 0]])
    }, dtype='Ragged[float32]'), dict(x='x', y='y', axis=1))
])
def test_area_to_zero_fixedrange(DataFrame, df_kwargs, cvs_kwargs):
    if DataFrame is dask_cudf_DataFrame:
        if df_kwargs.get('dtype', '').startswith('Ragged'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-2.25, 2.25), 5), 5)

    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-3.75, 3.75), 9), 9)

    cvs = ds.Canvas(plot_width=9, plot_height=5,
                    x_range=[-3.75, 3.75], y_range=[-2.25, 2.25])

    ddf = DataFrame(**df_kwargs)

    agg = cvs.area(ddf, agg=ds.count(), **cvs_kwargs)

    sol = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0]],
                   dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [
    # axis1 none constant
    (dict(data={
        'x0': [-4, 0],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, 0],
        'y1': [-4, -4],
        'y2': [0, 0]
    }, dtype='float32'),
     dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis1 y constant
    (dict(data={
        'x0': [-4, 0],
        'x1': [-2, 2],
        'x2': [0, 4],
    }, dtype='float32'),
     dict(x=['x0', 'x1', 'x2'], y=np.array([0, -4, 0], dtype='float32'), axis=1)),

    # axis0 single
    (dict(data={
        'x': [-4, -2, 0, 0, 2, 4],
        'y': [0, -4, 0, 0, -4, 0],
    }), dict(x='x', y='y', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [0, 2, 4],
        'y0': [0, -4, 0],
        'y1': [0, -4, 0],
    }, dtype='float32'), dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)),

    # axis0 multi, y string
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [0, 2, 4],
        'y0': [0, -4, 0],
    }, dtype='float32'), dict(x=['x0', 'x1'], y='y0', axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': [[-4, -2, 0], [0, 2, 4]],
        'y': [[0, -4, 0], [0, -4, 0]]
    }, dtype='Ragged[float32]'), dict(x='x', y='y', axis=1))
])
def test_area_to_zero_autorange(DataFrame, df_kwargs, cvs_kwargs):
    if DataFrame is dask_cudf_DataFrame:
        if df_kwargs.get('dtype', '').startswith('Ragged'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 0.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    ddf = DataFrame(**df_kwargs)
    agg = cvs.area(ddf, agg=ds.count(), **cvs_kwargs)

    sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]],
                   dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [
    # axis1 none constant
    (dict(data={
        'x0': [-4, np.nan],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, np.nan],
        'y1': [-4, 4],
        'y2': [0, 0]
    }, dtype='float32'), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis0 single
    (dict(data={
        'x': [-4, -2, 0, np.nan, 2, 4],
        'y': [0, -4, 0, np.nan, 4, 0],
    }), dict(x='x', y='y', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [np.nan, 2, 4],
        'y0': [0, -4, 0],
        'y1': [np.nan, 4, 0],
    }, dtype='float32'), dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': [[-4, -2, 0], [2, 4]],
        'y': [[0, -4, 0], [4, 0]],
    }, dtype='Ragged[float32]'), dict(x='x', y='y', axis=1))
])
def test_area_to_zero_autorange_gap(DataFrame, df_kwargs, cvs_kwargs):
    if DataFrame is dask_cudf_DataFrame:
        if df_kwargs.get('dtype', '').startswith('Ragged'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    ddf = DataFrame(**df_kwargs)

    agg = cvs.area(ddf, agg=ds.count(), **cvs_kwargs)

    sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
                   dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [
    # axis1 none constant
    (dict(data={
        'x0': [-4, 0],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, 0],
        'y1': [-4, -4],
        'y2': [0, 0],
        'y3': [0, 0],
        'y4': [-2, -2],
        'y5': [0, 0],
    }, dtype='float32'),
     dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'],
          y_stack=['y3', 'y4', 'y5'], axis=1)),

    # axis1 y constant
    (dict(data={
        'x0': [-4, 0],
        'x1': [-2, 2],
        'x2': [0, 4],
    }, dtype='float32'),
     dict(x=['x0', 'x1', 'x2'], y=np.array([0, -4, 0]),
          y_stack=np.array([0, -2, 0], dtype='float32'), axis=1)),

    # axis0 single
    (dict(data={
        'x': [-4, -2, 0, 0, 2, 4],
        'y': [0, -4, 0, 0, -4, 0],
        'y_stack': [0, -2, 0, 0, -2, 0],
    }), dict(x='x', y='y', y_stack='y_stack', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [0, 2, 4],
        'y0': [0, -4, 0],
        'y1': [0, -4, 0],
        'y2': [0, -2, 0],
        'y3': [0, -2, 0],
    }, dtype='float32'),
     dict(x=['x0', 'x1'], y=['y0', 'y1'], y_stack=['y2', 'y3'], axis=0)),

    # axis0 multi, y string
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [0, 2, 4],
        'y0': [0, -4, 0],
        'y2': [0, -2, 0],
    }, dtype='float32'), dict(x=['x0', 'x1'], y='y0', y_stack='y2', axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': [[-4, -2, 0], [0, 2, 4]],
        'y': [[0, -4, 0], [0, -4, 0]],
        'y_stack': [[0, -2, 0], [0, -2, 0]]
    }, dtype='Ragged[float32]'), dict(x='x', y='y', y_stack='y_stack', axis=1))
])
def test_area_to_line_autorange(DataFrame, df_kwargs, cvs_kwargs):
    if DataFrame is dask_cudf_DataFrame:
        if df_kwargs.get('dtype', '').startswith('Ragged'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 0.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    ddf = DataFrame(**df_kwargs)
    agg = cvs.area(ddf, agg=ds.count(), **cvs_kwargs)

    sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                   dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [
    # axis1 none constant
    (dict(data={
        'x0': [-4, np.nan],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, np.nan],
        'y1': [-4, 4],
        'y2': [0, 0],
        'y4': [0, 0],
        'y5': [0, 0],
        'y6': [0, 0]
    }, dtype='float32'),
     dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'],
          y_stack=['y4', 'y5', 'y6'], axis=1)),

    # axis0 single
    (dict(data={
        'x': [-4, -2, 0, np.nan, 2, 4],
        'y': [0, -4, 0, np.nan, 4, 0],
        'y_stack': [0, 0, 0, 0, 0, 0],
    }), dict(x='x', y='y', y_stack='y_stack', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [np.nan, 2, 4],
        'y0': [0, -4, 0],
        'y1': [np.nan, 4, 0],
        'y2': [0, 0, 0],
        'y3': [0, 0, 0],
    }, dtype='float32'),
     dict(x=['x0', 'x1'], y=['y0', 'y1'], y_stack=['y2', 'y3'], axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': [[-4, -2, 0], [2, 4]],
        'y': [[0, -4, 0], [4, 0]],
        'y_stack': [[0, 0, 0], [0, 0]],
    }, dtype='Ragged[float32]'), dict(x='x', y='y', y_stack='y_stack', axis=1))
])
def test_area_to_line_autorange_gap(DataFrame, df_kwargs, cvs_kwargs):
    if DataFrame is dask_cudf_DataFrame:
        if df_kwargs.get('dtype', '').startswith('Ragged'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    ddf = DataFrame(**df_kwargs)

    # When a line is specified to fill to, this line is not included in
    # the fill.  So we expect the y=0 line to not be filled.
    agg = cvs.area(ddf, agg=ds.count(), **cvs_kwargs)

    sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
                   dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


def test_trimesh_no_double_edge():
    """Assert that when two triangles share an edge that would normally get
    double-drawn, the edge is only drawn for the rightmost (or bottommost)
    triangle.
    """
    import multiprocessing as mp
    # Test left/right edge shared
    verts = dd.from_pandas(pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                                         'y': [4, 5, 5, 5, 4, 4]}), npartitions=mp.cpu_count())
    tris = dd.from_pandas(pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [1, 2]}), npartitions=mp.cpu_count())
    # Plot dims and x/y ranges need to be set such that the edge is drawn twice:
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)


@pytest.mark.parametrize('npartitions', list(range(1, 6)))
def test_trimesh_dask_partitions(npartitions):
    """Assert that when two triangles share an edge that would normally get
    double-drawn, the edge is only drawn for the rightmost (or bottommost)
    triangle.
    """
    # Test left/right edge shared
    verts = dd.from_pandas(pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                                         'y': [4, 5, 5, 5, 4, 4]}),
                           npartitions=npartitions)
    tris = dd.from_pandas(
        pd.DataFrame(
            {'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [1, 2]}),
        npartitions=npartitions)

    cvs = ds.Canvas(plot_width=20, plot_height=20,
                    x_range=(0, 5), y_range=(0, 5))

    # Precompute mesh with dask dataframes
    mesh = du.mesh(verts, tris)

    # Make sure mesh is a dask DataFrame
    assert isinstance(mesh, dd.DataFrame)

    # Check mesh length
    n = len(mesh)
    assert n == 6

    # Make sure we have expected number of partitions
    expected_chunksize = int(np.ceil(len(mesh) / (3*npartitions)) * 3)
    expected_npartitions = int(np.ceil(n / expected_chunksize))
    assert expected_npartitions == mesh.npartitions

    # Make sure triangles don't straddle partitions
    partitions_lens = mesh.map_partitions(len).compute()
    for partitions_len in partitions_lens:
        assert partitions_len % 3 == 0

    agg = cvs.trimesh(verts, tris, mesh)
    sol = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(
        np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)
