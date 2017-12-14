import numpy as np
import pandas as pd
import xarray as xr

import datashader as ds

import pytest


df = pd.DataFrame({'x': np.array(([0.] * 10 + [1] * 10)),
                   'y': np.array(([0.] * 5 + [1] * 5 + [0] * 5 + [1] * 5)),
                   'log_x': np.array(([1.] * 10 + [10] * 10)),
                   'log_y': np.array(([1.] * 5 + [10] * 5 + [1] * 5 + [10] * 5)),
                   'i32': np.arange(20, dtype='i4'),
                   'i64': np.arange(20, dtype='i8'),
                   'f32': np.arange(20, dtype='f4'),
                   'f64': np.arange(20, dtype='f8'),
                   'empty_bin': np.array([0.] * 15 + [np.nan] * 5),
                   'cat': ['a']*5 + ['b']*5 + ['c']*5 + ['d']*5})
df.cat = df.cat.astype('category')
df.f32[2] = np.nan
df.f64[2] = np.nan

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


def test_count():
    out = xr.DataArray(np.array([[5, 5], [5, 5]], dtype='i4'),
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.count('i32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.count('i64')), out)
    assert_eq(c.points(df, 'x', 'y', ds.count()), out)
    out = xr.DataArray(np.array([[4, 5], [5, 5]], dtype='i4'),
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.count('f32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.count('f64')), out)


def test_any():
    out = xr.DataArray(np.array([[True, True], [True, True]]),
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.any('i64')), out)
    assert_eq(c.points(df, 'x', 'y', ds.any('f64')), out)
    assert_eq(c.points(df, 'x', 'y', ds.any()), out)
    out = xr.DataArray(np.array([[True, True], [True, False]]),
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.any('empty_bin')), out)


def test_sum():
    out = xr.DataArray(df.i32.values.reshape((2, 2, 5)).sum(axis=2, dtype='f8').T,
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.sum('i32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.sum('i64')), out)
    out = xr.DataArray(np.nansum(df.f64.values.reshape((2, 2, 5)), axis=2).T,
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.sum('f32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.sum('f64')), out)


def test_min():
    out = xr.DataArray(df.i64.values.reshape((2, 2, 5)).min(axis=2).astype('f8').T,
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.min('i32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.min('i64')), out)
    assert_eq(c.points(df, 'x', 'y', ds.min('f32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.min('f64')), out)


def test_max():
    out = xr.DataArray(df.i64.values.reshape((2, 2, 5)).max(axis=2).astype('f8').T,
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.max('i32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.max('i64')), out)
    assert_eq(c.points(df, 'x', 'y', ds.max('f32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.max('f64')), out)


def test_mean():
    out = xr.DataArray(df.i32.values.reshape((2, 2, 5)).mean(axis=2, dtype='f8').T,
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.mean('i32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.mean('i64')), out)
    out = xr.DataArray(np.nanmean(df.f64.values.reshape((2, 2, 5)), axis=2).T,
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.mean('f32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.mean('f64')), out)


def test_var():
    out = xr.DataArray(df.i32.values.reshape((2, 2, 5)).var(axis=2, dtype='f8').T,
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.var('i32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.var('i64')), out)
    out = xr.DataArray(np.nanvar(df.f64.values.reshape((2, 2, 5)), axis=2).T,
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.var('f32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.var('f64')), out)


def test_std():
    out = xr.DataArray(df.i32.values.reshape((2, 2, 5)).std(axis=2, dtype='f8').T,
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.std('i32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.std('i64')), out)
    out = xr.DataArray(np.nanstd(df.f64.values.reshape((2, 2, 5)), axis=2).T,
                       coords=coords, dims=dims)
    assert_eq(c.points(df, 'x', 'y', ds.std('f32')), out)
    assert_eq(c.points(df, 'x', 'y', ds.std('f64')), out)


def test_count_cat():
    sol = np.array([[[5, 0, 0, 0],
                     [0, 0, 5, 0]],
                    [[0, 5, 0, 0],
                     [0, 0, 0, 5]]])
    out = xr.DataArray(sol, coords=(coords + [['a', 'b', 'c', 'd']]),
                       dims=(dims + ['cat']))
    agg = c.points(df, 'x', 'y', ds.count_cat('cat'))
    assert_eq(agg, out)


def test_multiple_aggregates():
    agg = c.points(df, 'x', 'y',
                   ds.summary(f64_std=ds.std('f64'),
                              f64_mean=ds.mean('f64'),
                              i32_sum=ds.sum('i32'),
                              i32_count=ds.count('i32')))

    f = lambda x: xr.DataArray(x, coords=coords, dims=dims)
    assert_eq(agg.f64_std, f(np.nanstd(df.f64.values.reshape((2, 2, 5)), axis=2).T))
    assert_eq(agg.f64_mean, f(np.nanmean(df.f64.values.reshape((2, 2, 5)), axis=2).T))
    assert_eq(agg.i32_sum, f(df.i32.values.reshape((2, 2, 5)).sum(axis=2, dtype='f8').T))
    assert_eq(agg.i32_count, f(np.array([[5, 5], [5, 5]], dtype='i4')))


def test_auto_range_points():
    n = 10
    data = np.arange(n, dtype='i4')
    df = pd.DataFrame({'time': np.arange(n),
                       'x': data,
                       'y': data})

    cvs = ds.Canvas(plot_width=n, plot_height=n)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((n, n), int)
    np.fill_diagonal(sol, 1)
    np.testing.assert_equal(agg.data, sol)

    cvs = ds.Canvas(plot_width=n+1, plot_height=n+1)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((n+1, n+1), int)
    np.fill_diagonal(sol, 1)
    sol[5, 5] = 0
    np.testing.assert_equal(agg.data, sol)

    n = 4
    data = np.arange(n, dtype='i4')
    df = pd.DataFrame({'time': np.arange(n),
                       'x': data,
                       'y': data})

    cvs = ds.Canvas(plot_width=2*n, plot_height=2*n)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((2*n, 2*n), int)
    np.fill_diagonal(sol, 1)
    sol[[range(1, 4, 2)]] = 0
    sol[[range(4, 8, 2)]] = 0
    np.testing.assert_equal(agg.data, sol)

    cvs = ds.Canvas(plot_width=2*n+1, plot_height=2*n+1)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((2*n+1, 2*n+1), int)
    sol[0, 0] = 1
    sol[3, 3] = 1
    sol[6, 6] = 1
    sol[8, 8] = 1
    np.testing.assert_equal(agg.data, sol)


def test_uniform_points():
    n = 101
    df = pd.DataFrame({'time': np.ones(2*n, dtype='i4'),
                       'x': np.concatenate((np.arange(n, dtype='f8'),
                                            np.arange(n, dtype='f8'))),
                       'y': np.concatenate(([0.] * n, [1.] * n))})

    cvs = ds.Canvas(plot_width=10, plot_height=2, y_range=(0, 1))
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.array([[10] * 9 + [11], [10] * 9 + [11]], dtype='i4')
    np.testing.assert_equal(agg.data, sol)


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


def test_log_axis_points():
    axis = ds.core.LogAxis()
    logcoords = axis.compute_index(axis.compute_scale_and_translate((1, 10), 2), 2)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)

    sol = np.array([[5, 5], [5, 5]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, logcoords],
                       dims=['y', 'log_x'])
    assert_eq(c_logx.points(df, 'log_x', 'y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, lincoords],
                       dims=['log_y', 'x'])
    assert_eq(c_logy.points(df, 'x', 'log_y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, logcoords],
                       dims=['log_y', 'log_x'])
    assert_eq(c_logxy.points(df, 'log_x', 'log_y', ds.count('i32')), out)


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
    assert_eq(agg, out)


def test_log_axis_line():
    axis = ds.core.LogAxis()
    logcoords = axis.compute_index(axis.compute_scale_and_translate((1, 10), 2), 2)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)

    sol = np.array([[5, 5], [5, 5]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, logcoords],
                       dims=['y', 'log_x'])
    assert_eq(c_logx.line(df, 'log_x', 'y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, lincoords],
                       dims=['log_y', 'x'])
    assert_eq(c_logy.line(df, 'x', 'log_y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, logcoords],
                       dims=['log_y', 'log_x'])
    assert_eq(c_logxy.line(df, 'log_x', 'log_y', ds.count('i32')), out)


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
    assert_eq(agg, out)

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
