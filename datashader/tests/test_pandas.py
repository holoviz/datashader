from __future__ import absolute_import
from collections import OrderedDict
import os
from numpy import nan

import numpy as np
import pandas as pd
import xarray as xr

import datashader as ds

import pytest

from datashader.datatypes import RaggedDtype

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
df_pd.at[2,'f32'] = nan
df_pd.at[2,'f64'] = nan
dfs_pd = [df_pd]

if "DATASHADER_TEST_GPU" in os.environ:
    test_gpu = bool(int(os.environ["DATASHADER_TEST_GPU"]))
else:
    test_gpu = None


try:
    import spatialpandas as sp
    from spatialpandas.geometry import LineDtype
except ImportError:
    LineDtype = None
    sp = None


def pd_DataFrame(*args, **kwargs):
    if kwargs.pop("geo", False):
        return sp.GeoDataFrame(*args, **kwargs)
    else:
        return pd.DataFrame(*args, **kwargs)


try:
    import cudf
    import cupy

    if not test_gpu:
        # GPU testing disabled even though cudf/cupy are available
        raise ImportError

    def cudf_DataFrame(*args, **kwargs):
        assert not kwargs.pop("geo", False)
        return cudf.DataFrame.from_pandas(
            pd.DataFrame(*args, **kwargs), nan_as_null=False
        )
    df_cuda = cudf_DataFrame(df_pd)
    dfs = [df_pd, df_cuda]
    DataFrames = [pd_DataFrame, cudf_DataFrame]
except ImportError:
    cudf = cupy = None
    dfs = [df_pd]
    DataFrames = [pd_DataFrame]


c = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))
c_logx = ds.Canvas(plot_width=2, plot_height=2, x_range=(1, 10),
                   y_range=(0, 1), x_axis_type='log')
c_logy = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1),
                   y_range=(1, 10), y_axis_type='log')
c_logxy = ds.Canvas(plot_width=2, plot_height=2, x_range=(1, 10),
                    y_range=(1, 10), x_axis_type='log', y_axis_type='log')

axis = ds.core.LinearAxis()
lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)
coords = OrderedDict([('x', lincoords), ('y', lincoords)])
dims = ['y', 'x']


def assert_eq_xr(agg, b, close=False):
    """Assert that two xarray DataArrays are equal, handling the possibility
    that the two DataArrays may be backed by ndarrays of different types"""
    if cupy:
        if isinstance(agg.data, cupy.ndarray):
            agg = xr.DataArray(
                cupy.asnumpy(agg.data), coords=agg.coords, dims=agg.dims
            )
        if isinstance(b.data, cupy.ndarray):
            b = xr.DataArray(
                cupy.asnumpy(b.data), coords=b.coords, dims=b.dims
            )
    if close:
        xr.testing.assert_allclose(agg, b)
    else:
        xr.testing.assert_equal(agg, b)

def assert_eq_ndarray(data, b):
    """Assert that two ndarrays are equal, handling the possibility that the
    ndarrays are of different types"""
    if cupy and isinstance(data, cupy.ndarray):
        data = cupy.asnumpy(data)
    np.testing.assert_equal(data, b)


def floats(n):
    """Returns contiguous list of floats from initial point"""
    while True:
        yield n
        n = n + np.spacing(n)


def values(s):
    """Get numpy array of values from pandas-like Series, handling Series
    of different types"""
    if cudf and isinstance(s, cudf.Series):
        return s.to_array(fillna=np.nan)
    else:
        return s.values


def test_gpu_dependencies():
    if test_gpu is True and cudf is None:
        pytest.fail("cudf and/or cupy not available and DATASHADER_TEST_GPU=1")


@pytest.mark.parametrize('df', dfs)
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


@pytest.mark.parametrize('df', dfs)
def test_any(df):
    out = xr.DataArray(np.array([[True, True], [True, True]]),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.any('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.any('f64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.any()), out)
    out = xr.DataArray(np.array([[True, True], [True, False]]),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.any('empty_bin')), out)


@pytest.mark.parametrize('df', dfs)
def test_sum(df):
    out = xr.DataArray(values(df.i32).reshape((2, 2, 5)).sum(axis=2, dtype='f8').T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.sum('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.sum('i64')), out)
    out = xr.DataArray(np.nansum(values(df.f64).reshape((2, 2, 5)), axis=2).T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.sum('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.sum('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_min(df):
    out = xr.DataArray(values(df.i64).reshape((2, 2, 5)).min(axis=2).astype('f8').T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.min('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.min('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.min('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.min('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_max(df):
    out = xr.DataArray(values(df.i64).reshape((2, 2, 5)).max(axis=2).astype('f8').T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.max('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.max('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.max('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.max('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_mean(df):
    out = xr.DataArray(values(df.i32).reshape((2, 2, 5)).mean(axis=2, dtype='f8').T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.mean('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.mean('i64')), out)
    out = xr.DataArray(np.nanmean(values(df.f64).reshape((2, 2, 5)), axis=2).T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.mean('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.mean('f64')), out)


@pytest.mark.parametrize('df', [df_pd])
def test_var(df):
    out = xr.DataArray(values(df.i32).reshape((2, 2, 5)).var(axis=2, dtype='f8').T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.var('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.var('i64')), out)
    out = xr.DataArray(np.nanvar(values(df.f64).reshape((2, 2, 5)), axis=2).T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.var('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.var('f64')), out)


@pytest.mark.parametrize('df', [df_pd])
def test_std(df):
    out = xr.DataArray(values(df.i32).reshape((2, 2, 5)).std(axis=2, dtype='f8').T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.std('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.std('i64')), out)
    out = xr.DataArray(np.nanstd(values(df.f64).reshape((2, 2, 5)), axis=2).T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.std('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.std('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_count_cat(df):
    sol = np.array([[[5, 0, 0, 0],
                     [0, 0, 5, 0]],
                    [[0, 5, 0, 0],
                     [0, 0, 0, 5]]])
    out = xr.DataArray(
        sol,
        coords=OrderedDict(coords, cat=['a', 'b', 'c', 'd']),
        dims=(dims + ['cat']))
    agg = c.points(df, 'x', 'y', ds.count_cat('cat'))
    assert_eq_xr(agg, out)

@pytest.mark.parametrize('df', dfs)
def test_categorical_count(df):
    sol = np.array([[[5, 0, 0, 0],
                     [0, 0, 5, 0]],
                    [[0, 5, 0, 0],
                     [0, 0, 0, 5]]])
    out = xr.DataArray(
        sol,
        coords=OrderedDict(coords, cat=['a', 'b', 'c', 'd']),
        dims=(dims + ['cat']))
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.count('i32')))
    assert_eq_xr(agg, out)

@pytest.mark.parametrize('df', dfs)
def test_categorical_sum(df):
    sol = np.array([[[ 10, nan, nan, nan],
                     [nan, nan,  60, nan]],
                    [[nan,  35, nan, nan],
                     [nan, nan, nan,  85]]])
    out = xr.DataArray(
        sol,
        coords=OrderedDict(coords, cat=['a', 'b', 'c', 'd']),
        dims=(dims + ['cat']))
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.sum('i32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.sum('i64')))
    assert_eq_xr(agg, out)

    sol = np.array([[[8.0,  nan,  nan,  nan],
                     [nan,  nan, 60.0,  nan]],
                    [[nan, 35.0,  nan,  nan],
                     [nan,  nan,  nan, 85.0]]])
    out = xr.DataArray(
        sol,
        coords=OrderedDict(coords, cat=['a', 'b', 'c', 'd']),
        dims=(dims + ['cat']))
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.sum('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.sum('f64')))
    assert_eq_xr(agg, out)

@pytest.mark.parametrize('df', dfs)
def test_categorical_max(df):
    sol = np.array([[[  4, nan, nan, nan],
                     [nan, nan,  14, nan]],
                    [[nan,   9, nan, nan],
                     [nan, nan, nan,  19]]])
    out = xr.DataArray(
        sol,
        coords=OrderedDict(coords, cat=['a', 'b', 'c', 'd']),
        dims=(dims + ['cat']))
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.max('i32')))
    assert_eq_xr(agg, out)

@pytest.mark.parametrize('df', dfs)
def test_categorical_mean(df):
    sol = np.array([[[  2, nan, nan, nan],
                     [nan, nan,  12, nan]],
                    [[nan,   7, nan, nan],
                     [nan, nan, nan,  17]]])
    out = xr.DataArray(
        sol,
        coords=OrderedDict(coords, cat=['a', 'b', 'c', 'd']),
        dims=(dims + ['cat']))

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.mean('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.mean('f64')))
    assert_eq_xr(agg, out)

@pytest.mark.parametrize('df', dfs)
def test_categorical_var(df):
    if cudf and isinstance(df, cudf.DataFrame):
        pytest.skip(
            "The 'var' reduction is yet supported on the GPU"
        )

    sol = np.array([[[ 2.5,  nan,  nan,  nan],
                     [ nan,  nan,   2.,  nan]],
                    [[ nan,   2.,  nan,  nan],
                     [ nan,  nan,  nan,   2.]]])
    out = xr.DataArray(
        sol,
        coords=OrderedDict(coords, cat=['a', 'b', 'c', 'd']),
        dims=(dims + ['cat']))

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.var('f32')))
    assert_eq_xr(agg, out, True)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.var('f64')))
    assert_eq_xr(agg, out, True)

@pytest.mark.parametrize('df', dfs)
def test_categorical_std(df):
    if cudf and isinstance(df, cudf.DataFrame):
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
        coords=OrderedDict(coords, cat=['a', 'b', 'c', 'd']),
        dims=(dims + ['cat']))

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.std('f32')))
    assert_eq_xr(agg, out, True)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.std('f64')))
    assert_eq_xr(agg, out, True)

@pytest.mark.parametrize('df', dfs)
def test_multiple_aggregates(df):
    agg = c.points(df, 'x', 'y',
                   ds.summary(f64_mean=ds.mean('f64'),
                              i32_sum=ds.sum('i32'),
                              i32_count=ds.count('i32')))

    f = lambda x: xr.DataArray(x, coords=coords, dims=dims)
    assert_eq_xr(agg.f64_mean, f(np.nanmean(values(df.f64).reshape((2, 2, 5)), axis=2).T))
    assert_eq_xr(agg.i32_sum, f(values(df.i32).reshape((2, 2, 5)).sum(axis=2, dtype='f8').T))
    assert_eq_xr(agg.i32_count, f(np.array([[5, 5], [5, 5]], dtype='i4')))


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_auto_range_points(DataFrame):
    n = 10
    data = np.arange(n, dtype='i4')
    df = DataFrame({'time': np.arange(n),
                    'x': data,
                    'y': data})

    cvs = ds.Canvas(plot_width=n, plot_height=n)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((n, n), int)
    np.fill_diagonal(sol, 1)
    assert_eq_ndarray(agg.data, sol)

    cvs = ds.Canvas(plot_width=n+1, plot_height=n+1)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((n+1, n+1), int)
    np.fill_diagonal(sol, 1)
    sol[5, 5] = 0
    assert_eq_ndarray(agg.data, sol)

    n = 4
    data = np.arange(n, dtype='i4')
    df = DataFrame({'time': np.arange(n),
                    'x': data,
                    'y': data})

    cvs = ds.Canvas(plot_width=2*n, plot_height=2*n)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((2*n, 2*n), int)
    np.fill_diagonal(sol, 1)
    sol[np.array([tuple(range(1, 4, 2))])] = 0
    sol[np.array([tuple(range(4, 8, 2))])] = 0
    assert_eq_ndarray(agg.data, sol)

    cvs = ds.Canvas(plot_width=2*n+1, plot_height=2*n+1)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((2*n+1, 2*n+1), int)
    sol[0, 0] = 1
    sol[3, 3] = 1
    sol[6, 6] = 1
    sol[8, 8] = 1
    assert_eq_ndarray(agg.data, sol)


def test_uniform_points():
    n = 101
    df = pd.DataFrame({'time': np.ones(2*n, dtype='i4'),
                       'x': np.concatenate((np.arange(n, dtype='f8'),
                                            np.arange(n, dtype='f8'))),
                       'y': np.concatenate(([0.] * n, [1.] * n))})

    cvs = ds.Canvas(plot_width=10, plot_height=2, y_range=(0, 1))
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.array([[10] * 9 + [11], [10] * 9 + [11]], dtype='i4')
    assert_eq_ndarray(agg.data, sol)


@pytest.mark.parametrize('high', [9, 10, 99, 100])
@pytest.mark.parametrize('low', [0])
def test_uniform_diagonal_points(low, high):
    bounds = (low, high)
    x_range, y_range = bounds, bounds

    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    n = width * height
    df = pd.DataFrame({'time': np.ones(n, dtype='i4'),
                       'x': np.array([np.arange(*x_range, dtype='f8')] * width).flatten(),
                       'y': np.array([np.arange(*y_range, dtype='f8')] * height).flatten()})

    cvs = ds.Canvas(plot_width=2, plot_height=2, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))

    diagonal = agg.data.diagonal(0)
    assert sum(diagonal) == n
    assert abs(bounds[1] - bounds[0]) % 2 == abs(diagonal[1] / high - diagonal[0] / high)


@pytest.mark.parametrize('df', dfs)
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


@pytest.mark.skipif(not sp, reason="spatialpandas not installed")
def test_points_geometry_point():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0., 2.), 3), 3)

    df = sp.GeoDataFrame({
        'geom': pd.array(
            [[0, 0], [0, 1], [1, 1], [0, 2], [1, 2], [2, 2]], dtype='Point[float64]'),
        'v': [1, 2, 2, 3, 3, 3]
    })

    cvs = ds.Canvas(plot_width=3, plot_height=3)
    agg = cvs.points(df, geometry='geom', agg=ds.sum('v'))
    sol = np.array([[1, nan, nan],
                    [2, 2,   nan],
                    [3, 3,   3]], dtype='float64')
    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)

    # Aggregation should not have triggered calculation of spatial index
    assert df.geom.array._sindex is None

    # Generate spatial index and check that we get the same result
    df.geom.array.sindex
    agg = cvs.points(df, geometry='geom', agg=ds.sum('v'))
    assert_eq_xr(agg, out)


@pytest.mark.skipif(not sp, reason="spatialpandas not installed")
def test_points_geometry_multipoint():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0., 2.), 3), 3)

    df = sp.GeoDataFrame({
        'geom': pd.array(
            [[0, 0], [0, 1, 1, 1], [0, 2, 1, 2, 2, 2]], dtype='MultiPoint[float64]'),
        'v': [1, 2, 3]
    })

    cvs = ds.Canvas(plot_width=3, plot_height=3)
    agg = cvs.points(df, geometry='geom', agg=ds.sum('v'))
    sol = np.array([[1, nan, nan],
                    [2, 2,   nan],
                    [3, 3,   3]], dtype='float64')
    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)

    # Aggregation should not have triggered calculation of spatial index
    assert df.geom.array._sindex is None

    # Generate spatial index and check that we get the same result
    df.geom.array.sindex
    agg = cvs.points(df, geometry='geom', agg=ds.sum('v'))
    assert_eq_xr(agg, out)


def test_line():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-3., 3.), 7), 7)

    df = pd.DataFrame({'x': [4, 0, -4, -3, -2, -1.9, 0, 10, 10, 0, 4],
                       'y': [0, -4, 0, 1, 2, 2.1, 4, 20, 30, 4, 0]})
    cvs = ds.Canvas(plot_width=7, plot_height=7,
                    x_range=(-3, 3), y_range=(-3, 3))
    agg = cvs.line(df, 'x', 'y', ds.count())
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


def test_points_on_edge():
    df = pd.DataFrame(dict(x=[0, 0.5, 1.1, 1.5, 2.2, 3, 3, 0],
                           y=[0, 0, 0, 0, 0, 0, 3, 3]))

    canvas = ds.Canvas(plot_width=3, plot_height=3,
                       x_range=(0, 3), y_range=(0, 3))

    agg = canvas.points(df, 'x', 'y', agg=ds.count())

    sol = np.array([[2, 2, 2],
                    [0, 0, 0],
                    [1, 0, 1]], dtype='int32')
    out = xr.DataArray(sol,
                       coords=[('x', [0.5, 1.5, 2.5]),
                               ('y', [0.5, 1.5, 2.5])],
                       dims=['y', 'x'])

    assert_eq_xr(agg, out)


def test_lines_on_edge():
    df = pd.DataFrame(dict(x=[0, 3, 3, 0],
                           y=[0, 0, 3, 3]))

    canvas = ds.Canvas(plot_width=3, plot_height=3,
                       x_range=(0, 3), y_range=(0, 3))

    agg = canvas.line(df, 'x', 'y', agg=ds.count())

    sol = np.array([[1, 1, 1],
                    [0, 0, 1],
                    [1, 1, 1]], dtype='int32')
    out = xr.DataArray(sol,
                       coords=[('x', [0.5, 1.5, 2.5]),
                               ('y', [0.5, 1.5, 2.5])],
                       dims=['y', 'x'])

    assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs_pd)
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


def test_subpixel_line_start():
    cvs = ds.Canvas(plot_width=5, plot_height=5, x_range=(1, 3), y_range=(0, 1))

    df = pd.DataFrame(dict(x=[1, 2, 3], y0=[0.0, 0.0, 0.0], y1=[0.0, 0.08, 0.04]))
    agg = cvs.line(df, 'x', ['y0', 'y1'], agg=ds.count(), axis=1)
    xcoords = axis.compute_index(axis.compute_scale_and_translate((1., 3), 5), 5)
    ycoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 5), 5)
    sol = np.array([
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype='i4')
    out = xr.DataArray(
        sol, coords=[ycoords, xcoords], dims=['y', 'x']
    )
    assert_eq_xr(agg, out)


def test_auto_range_line():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-10., 10.), 5), 5)

    df = pd.DataFrame({'x': [-10,  0, 10,   0, -10],
                       'y': [  0, 10,  0, -10,   0]})
    cvs = ds.Canvas(plot_width=5, plot_height=5)
    agg = cvs.line(df, 'x', 'y', ds.count())
    sol = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [2, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.skipif(not sp, reason="spatialpandas not installed")
@pytest.mark.parametrize('geom_data,geom_type', [
    ([0, 0, 1, 1, 2, 0, 0, 0], 'line'),
    ([[0, 0, 1, 1, 2, 0, 0, 0]], 'multiline'),
    ([0, 0, 1, 1, 2, 0, 0, 0], 'ring'),
    ([[0, 0, 1, 1, 2, 0, 0, 0]], 'polygon'),
    ([[[0, 0, 1, 1, 2, 0, 0, 0]]], 'multipolygon'),
])
def test_closed_ring_line(geom_data, geom_type):
    gdf = sp.GeoDataFrame(
        {'geometry': sp.GeoSeries([geom_data], dtype=geom_type)}
    )
    cvs = ds.Canvas(plot_width=4, plot_height=4)
    agg = cvs.line(gdf, geometry='geometry', agg=ds.count())

    coords_x = axis.compute_index(axis.compute_scale_and_translate((0., 2), 4), 4)
    coords_y = axis.compute_index(axis.compute_scale_and_translate((0., 1), 4), 4)
    sol = np.array([
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
    ])
    out = xr.DataArray(sol, coords=[coords_y, coords_x], dims=['y', 'x'])

    if geom_type.endswith("line"):
        # Closed rings represented as line/multiLine arrays will double count the
        # starting pixel
        out[0, 0] = 2

    assert_eq_xr(agg, out)


def test_trimesh_no_double_edge():
    """Assert that when two triangles share an edge that would normally get
    double-drawn, the edge is only drawn for the rightmost (or bottommost)
    triangle.
    """
    # Test left/right edge shared
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [1, 2]})
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

    # Test top/bottom edge shared
    verts = pd.DataFrame({'x': [3, 3, 1, 1, 3, 3],
                          'y': [4, 1, 4, 4, 5, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [3, 1]})
    # Plot dims and x/y ranges need to be set such that the edge is drawn twice:
    cvs = ds.Canvas(plot_width=22, plot_height=22, x_range=(0, 10), y_range=(0, 10))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[10:20, :20], sol)

def test_trimesh_interp():
    """Assert that triangles are interpolated when vertex values are provided.
    """
    verts = pd.DataFrame({'x': [0, 5, 10],
                          'y': [0, 10, 0]})
    tris = pd.DataFrame({'v0': [0], 'v1': [1], 'v2': [2],
                         'val': [1]})
    cvs = ds.Canvas(plot_width=10, plot_height=10, x_range=(0, 10), y_range=(0, 10))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values), sol)

    verts = pd.DataFrame({'x': [0, 5, 10],
                          'y': [0, 10, 0],
                          'z': [1, 5, 3]})
    cvs = ds.Canvas(plot_width=10, plot_height=10, x_range=(0, 10), y_range=(0, 10))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 4, 4, 0, 0, 0],
        [0, 0, 0, 3, 3, 3, 3, 0, 0, 0],
        [0, 0, 0, 2, 3, 3, 3, 3, 0, 0],
        [0, 0, 2, 2, 2, 3, 3, 3, 0, 0],
        [0, 0, 2, 2, 2, 2, 2, 3, 3, 0],
        [0, 1, 1, 1, 2, 2, 2, 2, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values), sol)

def test_trimesh_simplex_weights():
    """Assert that weighting the simplices works as expected.
    """
    # val is float
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [2., 4.]}) # floats
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 4, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)

    # val is int
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [3, 4]})
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 4, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)

def test_trimesh_vertex_weights():
    """Assert that weighting the vertices works as expected.
    """
    # z is float
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4],
                          'z': [1., 1., 1., 2., 2., 2.]}, columns=['x', 'y', 'z'])
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5]})
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='f8')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0.).values)[:5], sol)

    # val is int
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4],
                          'val': [2, 2, 2, 3, 3, 3]}, columns=['x', 'y', 'val'])
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5]})
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)

def test_trimesh_winding_detect():
    """Assert that CCW windings get converted to CW.
    """
    # val is int, winding is CCW
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [2, 5], 'v2': [1, 4], 'val': [3, 4]})
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 4, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)

    # val is float, winding is CCW
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [2, 5], 'v2': [1, 4], 'val': [3., 4.]}) # floats
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 4, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)

def test_trimesh_mesharg():
    """Assert that the ``mesh`` argument results in the same rasterization,
    despite the ``vertices`` and ``simplices`` arguments changing.
    """
    # z is float
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4],
                          'z': [1., 1., 1., 2., 2., 2.]}, columns=['x', 'y', 'z'])
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5]})
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='f8')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0.).values)[:5], sol)

    mesh = ds.utils.mesh(verts, tris)
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts[:1], tris[:1], mesh=mesh)
    np.testing.assert_array_equal(np.flipud(agg.fillna(0.).values)[:5], sol)

def test_trimesh_agg_api():
    """Assert that the trimesh aggregation API properly handles weights on the simplices."""
    pts = pd.DataFrame({'x': [1, 3, 4, 3, 3],
                        'y': [2, 1, 2, 1, 4]},
                       columns=['x', 'y'])
    tris = pd.DataFrame({'n1': [4, 1],
                         'n2': [1, 4],
                         'n3': [2, 0],
                         'weight': [0.83231525, 1.3053126]},
                        columns=['n1', 'n2', 'n3', 'weight'])
    cvs = ds.Canvas(x_range=(0, 10), y_range=(0, 10))
    agg = cvs.trimesh(pts, tris, agg=ds.mean('weight'))
    assert agg.shape == (600, 600)


def test_bug_570():
    # See https://github.com/holoviz/datashader/issues/570
    df = pd.DataFrame({
        'Time': [1456353642.2053893, 1456353642.2917893],
        'data': [-59.4948743433377, 506.4847376716022],
    }, columns=['Time', 'data'])

    x_range = (1456323293.9859753, 1456374687.0009754)
    y_range = (-228.56721300380943, 460.4042291124646)

    cvs = ds.Canvas(x_range=x_range, y_range=y_range,
                    plot_height=300, plot_width=1000)
    agg = cvs.line(df, 'Time', 'data', agg=ds.count())

    # Check location of line
    yi, xi = np.where(agg.values == 1)
    assert np.array_equal(yi, np.arange(73, 300))
    assert np.array_equal(xi, np.array([590] * len(yi)))


# # Line tests
line_manual_range_params = [
    # axis1 none constant
    ([{
        'x0': [4, -4],
        'x1': [0,  0],
        'x2': [-4, 4],
        'y0': [0,  0],
        'y1': [-4, 4],
        'y2': [0,  0]
    }], dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis1 x constant
    ([{
        'y0': [0,  0],
        'y1': [-4, 4],
        'y2': [0,  0]
    }], dict(x=np.array([-4, 0, 4]), y=['y0', 'y1', 'y2'], axis=1)),

    # axis1 y constant
    ([{
        'x0': [0, 0],
        'x1': [-4, 4],
        'x2': [0, 0]
    }], dict(x=['x0', 'x1', 'x2'], y=np.array([-4, 0, 4]), axis=1)),

    # axis0 single
    ([{
        'x': [0, -4, 0, np.nan, 0,  4, 0],
        'y': [-4, 0, 4, np.nan, -4, 0, 4],
    }], dict(x='x', y='y', axis=0)),

    # axis0 multi
    ([{
        'x0': [0, -4, 0],
        'x1': [0,  4, 0],
        'y0': [-4, 0, 4],
        'y1': [-4, 0, 4],
    }], dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)),

    # axis0 multi with string
    ([{
        'x0': [0, -4, 0],
        'x1': [0,  4, 0],
        'y0': [-4, 0, 4],
        'y1': [-4, 0, 4],
    }], dict(x=['x0', 'x1'], y='y0', axis=0)),

    # axis1 ragged arrays
    ([{
        'x': pd.array([[4, 0], [0, -4, 0, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4], [-4, 0, 4, 0]], dtype='Ragged[float32]')
    }], dict(x='x', y='y', axis=1)),
]
if sp:
    line_manual_range_params.append(
        # geometry
        ([{
            'geom': pd.array(
                [[4, 0, 0, -4], [0, -4, -4, 0, 0, 4, 4, 0]], dtype='Line[float32]'
            ),
        }], dict(geometry='geom'))
    )
@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_args,cvs_kwargs', line_manual_range_params)
def test_line_manual_range(DataFrame, df_args, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if (isinstance(getattr(df_args[0].get('x', []), 'dtype', ''), RaggedDtype) or
                sp and isinstance(
                    getattr(df_args[0].get('geom', []), 'dtype', ''), LineDtype
                )
        ):
            pytest.skip("cudf DataFrames do not support extension types")

    df = DataFrame(geo='geometry' in cvs_kwargs, *df_args)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(
        axis.compute_scale_and_translate((-3., 3.), 7), 7)

    cvs = ds.Canvas(plot_width=7, plot_height=7,
                    x_range=(-3, 3), y_range=(-3, 3))

    agg = cvs.line(df, agg=ds.count(), **cvs_kwargs)

    sol = np.array([[0, 0, 1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 0]], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


line_autorange_params = [
    # axis1 none constant
    ([{
        'x0': [0,  0],
        'x1': [-4, 4],
        'x2': [0,  0],
        'y0': [-4, -4],
        'y1': [0,  0],
        'y2': [4,  4]
    }], dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis1 y constant
    ([{
        'x0': [0, 0],
        'x1': [-4, 4],
        'x2': [0, 0]
    }], dict(x=['x0', 'x1', 'x2'], y=np.array([-4, 0, 4]), axis=1)),

    # axis0 single
    ([{
        'x': [0, -4, 0, np.nan, 0,  4, 0],
        'y': [-4, 0, 4, np.nan, -4, 0, 4],
    }], dict(x='x', y='y', axis=0)),

    # axis0 multi
    ([{
        'x0': [0, -4, 0],
        'x1': [0,  4, 0],
        'y0': [-4, 0, 4],
        'y1': [-4, 0, 4],
    }], dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)),

    # axis0 multi with string
    ([{
        'x0': [0, -4, 0],
        'x1': [0,  4, 0],
        'y0': [-4, 0, 4],
        'y1': [-4, 0, 4],
    }], dict(x=['x0', 'x1'], y='y0', axis=0)),

    # axis1 ragged arrays
    ([{
        'x': pd.array([[0, -4, 0], [0,  4, 0]], dtype='Ragged[float32]'),
        'y': pd.array([[-4, 0, 4], [-4, 0, 4]], dtype='Ragged[float32]')
    }], dict(x='x', y='y', axis=1)),
]
if sp:
    line_autorange_params.append(
        # geometry
        ([{
            'geom': pd.array(
                [[0, -4, -4, 0, 0, 4], [0, -4,  4, 0, 0, 4]], dtype='Line[float32]'
            ),
        }], dict(geometry='geom'))
    )
@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_args,cvs_kwargs', line_autorange_params)
def test_line_autorange(DataFrame, df_args, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if (isinstance(getattr(df_args[0].get('x', []), 'dtype', ''), RaggedDtype) or
                sp and isinstance(
                    getattr(df_args[0].get('geom', []), 'dtype', ''), LineDtype
                )
        ):
            pytest.skip("cudf DataFrames do not support extension types")

    df = DataFrame(geo='geometry' in cvs_kwargs, *df_args)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 9), 9)

    cvs = ds.Canvas(plot_width=9, plot_height=9)

    agg = cvs.line(df, agg=ds.count(), **cvs_kwargs)

    sol = np.array([[0, 0, 0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0, 0, 0, 0]], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_line_autorange_axis1_x_constant(DataFrame):
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 9), 9)

    xs = np.array([-4, 0, 4])
    df = DataFrame({
        'y0': [0,  0],
        'y1': [-4, 4],
        'y2': [0,  0]
    })

    cvs = ds.Canvas(plot_width=9, plot_height=9)

    agg = cvs.line(df,
                   xs,
                   ['y0', 'y1', 'y2'],
                   ds.count(),
                   axis=1)

    sol = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [2, 0, 0, 0, 0, 0, 0, 0, 2],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


# Sum aggregate
@pytest.mark.parametrize('DataFrame', DataFrames)
def test_line_agg_sum_axis1_none_constant(DataFrame):
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-3., 3.), 7), 7)

    df = DataFrame({
        'x0': [4, -4],
        'x1': [0,  0],
        'x2': [-4, 4],
        'y0': [0,  0],
        'y1': [-4, 4],
        'y2': [0,  0],
        'v': [7, 9]
    })

    cvs = ds.Canvas(plot_width=7, plot_height=7,
                    x_range=(-3, 3), y_range=(-3, 3))

    agg = cvs.line(df,
                   ['x0', 'x1', 'x2'],
                   ['y0', 'y1', 'y2'],
                   ds.sum('v'),
                   axis=1)
    nan = np.nan
    sol = np.array([[nan, nan, 7,   nan, 7,   nan, nan],
                    [nan, 7,   nan, nan, nan, 7,   nan],
                    [7,   nan, nan, nan, nan, nan, 7],
                    [nan, nan, nan, nan, nan, nan, nan],
                    [9,   nan, nan, nan, nan, nan, 9],
                    [nan, 9,   nan, nan, nan, 9,   nan],
                    [nan, nan, 9,   nan, 9,   nan, nan]], dtype='float32')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


def test_line_autorange_axis1_ragged():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 9), 9)

    df = pd.DataFrame({
        'x': pd.array([[4, 0], [0, -4, 0, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4], [-4, 0, 4, 0]], dtype='Ragged[float32]')
    })

    cvs = ds.Canvas(plot_width=9, plot_height=9)

    agg = cvs.line(df,
                   'x',
                   'y',
                   ds.count(),
                   axis=1)

    sol = np.array([[0, 0, 0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 2],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='i4')

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
        'x0': [-4., -2., 0],
        'x1': [np.nan, 2, 4],
        'y0': [0, -4, 0],
        'y1': [np.nan, 4, 0],
    }, dtype='float32'), dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': pd.array([[-4, -2, 0], [2, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4, 0], [4, 0]], dtype='Ragged[float32]')
    }), dict(x='x', y='y', axis=1))
])
def test_area_to_zero_fixedrange(DataFrame, df_kwargs, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if isinstance(getattr(df_kwargs['data'].get('x', []), 'dtype', ''), RaggedDtype):
            pytest.skip("cudf DataFrames do not support extension types")

    df = DataFrame(**df_kwargs)

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-2.25, 2.25), 5), 5)

    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-3.75, 3.75), 9), 9)

    cvs = ds.Canvas(plot_width=9, plot_height=5,
                    x_range=[-3.75, 3.75], y_range=[-2.25, 2.25])

    agg = cvs.area(df, agg=ds.count(), **cvs_kwargs)

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
    }, dtype='float32'), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

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
        'x': pd.array([[-4, -2, 0], [0, 2, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4, 0], [0, -4, 0]], dtype='Ragged[float32]')
    }), dict(x='x', y='y', axis=1))
])
def test_area_to_zero_autorange(DataFrame, df_kwargs, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if isinstance(getattr(df_kwargs['data'].get('x', []), 'dtype', ''), RaggedDtype):
            pytest.skip("cudf DataFrames do not support extension types")

    df = DataFrame(**df_kwargs)

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 0.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    agg = cvs.area(df, agg=ds.count(), **cvs_kwargs)

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
        # 'x0': [-4, 0],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, np.nan],
        # 'y0': [0, 1],
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
        'x': pd.array([[-4, -2, 0], [2, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4, 0], [4, 0]], dtype='Ragged[float32]')
    }), dict(x='x', y='y', axis=1))
])
def test_area_to_zero_autorange_gap(DataFrame, df_kwargs, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if isinstance(getattr(df_kwargs['data'].get('x', []), 'dtype', ''), RaggedDtype):
            pytest.skip("cudf DataFrames do not support extension types")

    df = DataFrame(**df_kwargs)

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    agg = cvs.area(df, agg=ds.count(), **cvs_kwargs)

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
    }, dtype='float32'), dict(x=['x0', 'x1'], y=['y0', 'y1'],
                              y_stack=['y2', 'y3'], axis=0)),

    # axis0 multi, y string
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [0, 2, 4],
        'y0': [0, -4, 0],
        'y2': [0, -2, 0],
    }, dtype='float32'), dict(x=['x0', 'x1'], y='y0', y_stack='y2', axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': pd.array([[-4, -2, 0], [0, 2, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4, 0], [0, -4, 0]], dtype='Ragged[float32]'),
        'y_stack': pd.array([[0, -2, 0], [0, -2, 0]], dtype='Ragged[float32]')
    }), dict(x='x', y='y', y_stack='y_stack', axis=1))
])
def test_area_to_line_autorange(DataFrame, df_kwargs, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if isinstance(getattr(df_kwargs['data'].get('x', []), 'dtype', ''), RaggedDtype):
            pytest.skip("cudf DataFrames do not support extension types")

    df = DataFrame(**df_kwargs)

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 0.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    agg = cvs.area(df, agg=ds.count(), **cvs_kwargs)

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


def test_area_to_line_autorange_gap():
    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    df = pd.DataFrame({
        'x': [-4, -2, 0, np.nan, 2, 4],
        'y0': [0, -4, 0, np.nan, 4, 0],
        'y1': [0,  0, 0, np.nan, 0, 0],
    })

    # When a line is specified to fill to, this line is not included in
    # the fill.  So we expect the y=0 line to not be filled.
    agg = cvs.area(df, 'x', 'y0', ds.count(), y_stack='y1')

    sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
                   dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x],
                       dims=['y0', 'x'])
    assert_eq_xr(agg, out)
