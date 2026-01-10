from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
import pytest
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

from datashader.resampling import infer_interval_breaks, infer_interval_breaks_2d
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
from datashader.tests.utils import dask_skip

array_modules = [np]

try:
    import dask
    import dask.array
    dask.config.set(scheduler='single-threaded')
    array_modules.append(dask.array)
except ImportError:
    class dask:
        array = None

try:
    import cudf
    import cupy
    array_modules.append(pytest.param(cupy, marks=pytest.mark.gpu))
except ImportError:
    cudf = None
    cupy = None


# Raster
@pytest.mark.parametrize('array_module', array_modules)
def test_raster_quadmesh_autorange_downsample(array_module):
    c = ds.Canvas(plot_width=4, plot_height=2)
    da = xr.DataArray(
        array_module.array(
            [[1,   2,  3,  4,  5,  6,  7,  8],
             [9,  10, 11, 12, 13, 14, 15, 16],
             [17, 18, 19, 20, 21, 22, 23, 24],
             [25, 26, 27, 28, 29, 30, 31, 32]]
        ),
        coords=[('b', [1, 2, 3, 4]),
                ('a', [1, 2, 3, 4, 5, 6, 7, 8])],
        name='Z')

    y_coords = np.linspace(1.5, 3.5, 2)
    x_coords = np.linspace(1.5, 7.5, 4)
    out = xr.DataArray(array_module.array(
        [[1+2+9+10, 3+4+11+12, 5+6+13+14, 7+8+15+16],
         [17+18+25+26., 19+20+27+28, 21+22+29+30, 23+24+31+32]],
        dtype='f8'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)


@pytest.mark.parametrize('array_module', array_modules)
def test_raster_quadmesh_autorange(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4)
    da = xr.DataArray(
        array_module.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]]
        ),
        coords=[('b', [1, 2]),
                ('a', [1, 2, 3, 4])],
        name='Z')

    y_coords = np.linspace(0.75, 2.25, 4)
    x_coords = np.linspace(0.75, 4.25, 8)
    out = xr.DataArray(array_module.array(
        [[1., 1., 2., 2., 3., 3., 4., 4.],
         [1., 1., 2., 2., 3., 3., 4., 4.],
         [5., 5., 6., 6., 7., 7., 8., 8.],
         [5., 5., 6., 6., 7., 7., 8., 8.]],
        dtype='f8'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (0.5, 4.5), close=True)
    assert_eq_ndarray(res.y_range, (0.5, 2.5), close=True)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (0.5, 4.5), close=True)
    assert_eq_ndarray(res.y_range, (0.5, 2.5), close=True)


@dask_skip
def test_raster_quadmesh_autorange_chunked():
    c = ds.Canvas(plot_width=8, plot_height=6)
    da = xr.DataArray(
        np.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12]]
        ),
        coords=[('b', [1, 2, 3]),
                ('a', [1, 2, 3, 4])],
        name='Z').chunk({'a': 2, 'b': 2})

    y_coords = np.linspace(0.75, 3.25, 6)
    x_coords = np.linspace(0.75, 4.25, 8)
    out = xr.DataArray(np.array(
        [[1., 1., 2., 2., 3., 3., 4., 4.],
         [1., 1., 2., 2., 3., 3., 4., 4.],
         [5., 5., 6., 6., 7., 7., 8., 8.],
         [5., 5., 6., 6., 7., 7., 8., 8.],
         [9., 9., 10., 10., 11., 11., 12., 12.],
         [9., 9., 10., 10., 11., 11., 12., 12.]],
        dtype='f8'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)


@pytest.mark.parametrize('array_module', array_modules)
def test_raster_quadmesh_upsampley_and_downsamplex(array_module):
    c = ds.Canvas(plot_width=2, plot_height=4)
    da = xr.DataArray(
        array_module.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]]
        ),
        coords=[('b', [1, 2]),
                ('a', [1, 2, 3, 4])],
        name='Z')

    y_coords = np.linspace(0.75, 2.25, 4)
    x_coords = np.linspace(1.5, 3.5, 2)
    out = xr.DataArray(array_module.array(
        [[3., 7.],
         [3., 7.],
         [11., 15.],
         [11., 15.]],
        dtype='f8'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)


@pytest.mark.parametrize('array_module', array_modules)
def test_raster_quadmesh_upsamplex_and_downsampley(array_module):
    c = ds.Canvas(plot_width=4, plot_height=2)
    da = xr.DataArray(
        array_module.array(
            [[1, 2],
             [3, 4],
             [5, 6],
             [7, 8]]
        ),
        coords=[('b', [1, 2, 3, 4]),
                ('a', [1, 2])],
        name='Z')

    x_coords = np.linspace(0.75, 2.25, 4)
    y_coords = np.linspace(1.5, 3.5, 2)
    out = xr.DataArray(array_module.array(
        [[4., 4., 6., 6.],
         [12., 12., 14., 14.]],
        dtype='f8'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)


@pytest.mark.parametrize('array_module', array_modules)
def test_raster_quadmesh_autorange_reversed(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4)
    da = xr.DataArray(
        array_module.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]]
        ),
        coords=[('b', [-1, -2]),
                ('a', [-1, -2, -3, -4])],
        name='Z')

    y_coords = np.linspace(-2.25, -0.75, 4)
    x_coords = np.linspace(-4.25, -0.75, 8)
    out = xr.DataArray(array_module.array(
        [[8., 8., 7., 7., 6., 6., 5., 5.],
         [8., 8., 7., 7., 6., 6., 5., 5.],
         [4., 4., 3., 3., 2., 2., 1., 1.],
         [4., 4., 3., 3., 2., 2., 1., 1.]],
        dtype='f8'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)


@pytest.mark.parametrize('array_module', array_modules)
def test_raster_quadmesh_manual_range(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4,
                  x_range=[1, 3],
                  y_range=[-1, 3])

    da = xr.DataArray(
        array_module.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]]
        ),
        coords=[('b', [1, 2]),
                ('a', [1, 2, 3, 4])],
        name='Z')

    y_coords = np.linspace(-0.5, 2.5, 4)
    x_coords = np.linspace(1.125, 2.875, 8)
    out = xr.DataArray(array_module.array(
        [[nan, nan, nan, nan, nan, nan, nan, nan],
         [1., 1., 2., 2., 2., 2., 3., 3.],
         [5., 5., 6., 6., 6., 6., 7., 7.],
         [nan, nan, nan, nan, nan, nan, nan, nan]],
        dtype='f8'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (1, 3), close=True)
    assert_eq_ndarray(res.y_range, (-1, 3), close=True)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (1, 3), close=True)
    assert_eq_ndarray(res.y_range, (-1, 3), close=True)


@pytest.mark.parametrize('array_module', array_modules)
def test_raster_subpixel_quads_represented(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4,
                  x_range=[0.5, 16.5],
                  y_range=[0.5, 8.5])

    da = xr.DataArray(
        array_module.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]]
        ),
        coords=[('b', [1, 2]),
                ('a', [1, 2, 3, 4])],
        name='Z')

    y_coords = np.linspace(1.5, 7.5, 4)
    x_coords = np.linspace(1.5, 15.5, 8)
    out = xr.DataArray(array_module.array(
        [[14., 22., nan, nan, nan, nan, nan, nan],
         [nan, nan, nan, nan, nan, nan, nan, nan],
         [nan, nan, nan, nan, nan, nan, nan, nan],
         [nan, nan, nan, nan, nan, nan, nan, nan]],
        dtype='f4'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)


# Rectilinear
@pytest.mark.parametrize('array_module', array_modules)
def test_rectilinear_quadmesh_autorange(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4)
    da = xr.DataArray(
        array_module.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]]
        ),
        coords=[('b', [1, 2]),
                ('a', [1, 2, 3, 8])],
        name='Z')

    y_coords = np.linspace(0.75, 2.25, 4)
    x_coords = np.linspace(1.125, 9.875, 8)
    out = xr.DataArray(array_module.array(
        [[3., 3., 3., 3., 4., 4., 4., 4.],
         [3., 3., 3., 3., 4., 4., 4., 4.],
         [11., 7., 7., 7., 8., 8., 8., 8.],
         [11., 7., 7., 7., 8., 8., 8., 8.]],
        dtype='f8'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (0.5, 10.5), close=True)
    assert_eq_ndarray(res.y_range, (0.5, 2.5), close=True)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (0.5, 10.5), close=True)
    assert_eq_ndarray(res.y_range, (0.5, 2.5), close=True)


@dask_skip
def test_rectilinear_quadmesh_autorange_chunked():
    c = ds.Canvas(plot_width=8, plot_height=6)
    da = xr.DataArray(
        np.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12]]
        ),
        coords=[('b', [1, 2, 3]),
                ('a', [1, 2, 3, 8])],
        name='Z').chunk({'a': 2, 'b': 3})

    y_coords = np.linspace(0.75, 3.25, 6)
    x_coords = np.linspace(1.125, 9.875, 8)
    out = xr.DataArray(np.array(
        [[3., 3., 3., 3., 4., 4., 4., 4.],
         [3., 3., 3., 3., 4., 4., 4., 4.],
         [11., 7., 7., 7., 8., 8., 8., 8.],
         [11., 7., 7., 7., 8., 8., 8., 8.],
         [19., 11., 11., 11., 12., 12., 12., 12.],
         [19., 11., 11., 11., 12., 12., 12., 12.]],
        dtype='f8'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (0.5, 10.5), close=True)
    assert_eq_ndarray(res.y_range, (0.5, 3.5), close=True)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (0.5, 10.5), close=True)
    assert_eq_ndarray(res.y_range, (0.5, 3.5), close=True)


@pytest.mark.parametrize('array_module', array_modules)
def test_rect_quadmesh_autorange_reversed(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4)
    da = xr.DataArray(
        array_module.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]]
        ),
        coords=[('b', [-1, -2]),
                ('a', [-1, -2, -3, -8])],
        name='Z')

    y_coords = np.linspace(-2.25, -0.75, 4)
    x_coords = np.linspace(-9.875, -1.125, 8)
    out = xr.DataArray(array_module.array(
        [[8., 8., 8., 8., 7., 7., 6., 5.],
         [8., 8., 8., 8., 7., 7., 6., 5.],
         [4., 4., 4., 4., 3., 3., 2., 1.],
         [4., 4., 4., 4., 3., 3., 2., 1.]],
        dtype='f8'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (-10.5, -0.5), close=True)
    assert_eq_ndarray(res.y_range, (-2.5, -0.5), close=True)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (-10.5, -0.5), close=True)
    assert_eq_ndarray(res.y_range, (-2.5, -0.5), close=True)


@pytest.mark.parametrize('array_module', array_modules)
def test_rect_quadmesh_manual_range(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4,
                  x_range=[1, 3],
                  y_range=[-1, 3])

    da = xr.DataArray(
        array_module.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]]
        ),
        coords=[('b', [1, 2]),
                ('a', [1, 2, 3, 8])],
        name='Z')

    y_coords = np.linspace(-0.5, 2.5, 4)
    x_coords = np.linspace(1.125, 2.875, 8)
    out = xr.DataArray(array_module.array(
        [[nan, nan, nan, nan, nan, nan, nan, nan],
         [1., 1., 2., 2., 2., 2., 3., 3.],
         [5., 5., 6., 6., 6., 6., 7., 7.],
         [nan, nan, nan, nan, nan, nan, nan, nan]],
        dtype='f8'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (1, 3), close=True)
    assert_eq_ndarray(res.y_range, (-1, 3), close=True)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (1, 3), close=True)
    assert_eq_ndarray(res.y_range, (-1, 3), close=True)


@pytest.mark.parametrize('array_module', array_modules)
def test_rect_subpixel_quads_represented(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4,
                  x_range=[0, 16],
                  y_range=[0, 8])

    da = xr.DataArray(
        array_module.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]]
        ),
        coords=[('b', [1, 2]),
                ('a', [1, 2.5, 3, 4])],
        name='Z')

    y_coords = np.linspace(1, 7, 4)
    x_coords = np.linspace(1, 15, 8)
    out = xr.DataArray(array_module.array(
        [[14., 22., nan, nan, nan, nan, nan, nan],
         [nan, nan, nan, nan, nan, nan, nan, nan],
         [nan, nan, nan, nan, nan, nan, nan, nan],
         [nan, nan, nan, nan, nan, nan, nan, nan]],
        dtype='f4'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)


# Curvilinear
@pytest.mark.parametrize('array_module', array_modules)
def test_curve_quadmesh_rect_autorange(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4)
    coord_array = dask.array if array_module is dask.array else np

    Qx = coord_array.array(
        [[1, 2],
         [1, 2]]
    )
    Qy = coord_array.array(
        [[1, 1],
         [2, 2]]
    )
    Z = np.arange(4, dtype='int32').reshape(2, 2)
    da = xr.DataArray(
        array_module.array(Z),
        coords={'Qx': (['Y', 'X'], Qx),
                'Qy': (['Y', 'X'], Qy)},
        dims=['Y', 'X'],
        name='Z',
    )

    y_coords = np.linspace(0.75, 2.25, 4)
    x_coords = np.linspace(0.625, 2.375, 8)
    out = xr.DataArray(array_module.array(
        [[0., 0., 0., 0., 1., 1., 1., 1.],
         [0., 0., 0., 0., 1., 1., 1., 1.],
         [2., 2., 2., 2., 3., 3., 3., 3.],
         [2., 2., 2., 2., 3., 3., 3., 3.]],
        dtype='f8'
        ),
        coords=[('Qy', y_coords),
                ('Qx', x_coords)]
    )

    res = c.quadmesh(da, x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)

    res = c.quadmesh(da.transpose('X', 'Y', transpose_coords=True), x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)


@pytest.mark.parametrize('array_module', array_modules)
def test_curve_quadmesh_autorange(array_module):
    c = ds.Canvas(plot_width=4, plot_height=8)
    coord_array = dask.array if array_module is dask.array else np

    Qx = coord_array.array(
        [[1, 2],
         [1, 2]]
    )
    Qy = coord_array.array(
        [[1, 1],
         [4, 2]]
    )
    Z = np.arange(4, dtype='int32').reshape(2, 2)
    da = xr.DataArray(
        array_module.array(Z),
        coords={'Qx': (['Y', 'X'], Qx),
                'Qy': (['Y', 'X'], Qy)},
        dims=['Y', 'X'],
        name='Z',
    )

    x_coords = np.linspace(0.75, 2.25, 4)
    y_coords = np.linspace(-0.5, 6.5, 8)
    out = xr.DataArray(array_module.array(
        [[nan, nan, nan, nan],
         [0.,  0.,  nan, nan],
         [0.,  0.,  1.,  1.],
         [0.,  0.,  3.,  3.],
         [2.,  2.,  3.,  nan],
         [2.,  2.,  nan, nan],
         [2.,  2.,  nan, nan],
         [2.,  nan, nan, nan]]
        ),
        coords=dict([
            ('Qx', x_coords),
            ('Qy', y_coords)]),
        dims=['Qy', 'Qx']
    )

    res = c.quadmesh(da, x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (0.5, 2.5), close=True)
    assert_eq_ndarray(res.y_range, (-1, 7), close=True)

    res = c.quadmesh(da.transpose('X', 'Y', transpose_coords=True), x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (0.5, 2.5), close=True)
    assert_eq_ndarray(res.y_range, (-1, 7), close=True)

@dask_skip
def test_curve_quadmesh_autorange_chunked():
    c = ds.Canvas(plot_width=4, plot_height=8)

    Qx = np.array(
        [[1, 2],
         [1, 2]]
    )
    Qy = np.array(
        [[1, 1],
         [4, 2]]
    )
    Z = np.arange(4, dtype='int32').reshape(2, 2)
    da = xr.DataArray(
        np.array(Z),
        coords={'Qx': (['Y', 'X'], Qx),
                'Qy': (['Y', 'X'], Qy)},
        dims=['Y', 'X'],
        name='Z',
    ).chunk({'X': 2, 'Y': 1})

    x_coords = np.linspace(0.75, 2.25, 4)
    y_coords = np.linspace(-0.5, 6.5, 8)
    out = xr.DataArray(np.array(
        [[nan, nan, nan, nan],
         [0.,  0.,  nan, nan],
         [0.,  0.,  1.,  1.],
         [0.,  0.,  3.,  3.],
         [2.,  2.,  3.,  nan],
         [2.,  2.,  nan, nan],
         [2.,  2.,  nan, nan],
         [2.,  nan, nan, nan]]
        ),
        coords=dict([
            ('Qx', x_coords),
            ('Qy', y_coords)]),
        dims=['Qy', 'Qx']
    )

    res = c.quadmesh(da, x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (0.5, 2.5), close=True)
    assert_eq_ndarray(res.y_range, (-1, 7), close=True)

    res = c.quadmesh(da.transpose('X', 'Y', transpose_coords=True), x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (0.5, 2.5), close=True)
    assert_eq_ndarray(res.y_range, (-1, 7), close=True)


@pytest.mark.parametrize('array_module', array_modules)
def test_curve_quadmesh_manual_range(array_module):
    c = ds.Canvas(plot_width=4, plot_height=8, x_range=[1, 2], y_range=[1, 3])
    coord_array = dask.array if array_module is dask.array else np

    Qx = coord_array.array(
        [[1, 2],
         [1, 2]]
    )
    Qy = coord_array.array(
        [[1, 1],
         [4, 2]]
    )
    Z = np.arange(4, dtype='int32').reshape(2, 2)
    da = xr.DataArray(
        array_module.array(Z),
        coords={'Qx': (['Y', 'X'], Qx),
                'Qy': (['Y', 'X'], Qy)},
        dims=['Y', 'X'],
        name='Z',
    )

    x_coords = np.linspace(1.125, 1.875, 4)
    y_coords = np.linspace(1.125, 2.875, 8)
    out = xr.DataArray(array_module.array(
        [[0., 0., 1., 1.],
         [0., 0., 1., 1.],
         [0., 0., 1., 1.],
         [0., 0., 1., 3.],
         [0., 0., 3., 3.],
         [0., 2., 3., 3.],
         [2., 2., 3., 3.],
         [2., 2., 3., 3.]]
        ),
        coords=dict([
            ('Qx', x_coords),
            ('Qy', y_coords)]),
        dims=['Qy', 'Qx']
    )

    res = c.quadmesh(da, x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (1, 2), close=True)
    assert_eq_ndarray(res.y_range, (1, 3), close=True)

    res = c.quadmesh(da.transpose('X', 'Y', transpose_coords=True), x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (1, 2), close=True)
    assert_eq_ndarray(res.y_range, (1, 3), close=True)


@pytest.mark.parametrize('array_module', array_modules)
def test_curve_quadmesh_manual_range_subpixel(array_module):
    c = ds.Canvas(plot_width=3, plot_height=5,
                  x_range=[-150, 150], y_range=[-250, 250])
    coord_array = dask.array if array_module is dask.array else np

    Qx = coord_array.array(
        [[1, 2],
         [1, 2]]
    )
    Qy = coord_array.array(
        [[1, 1],
         [4, 2]]
    )
    Z = np.arange(4, dtype='int32').reshape(2, 2)
    da = xr.DataArray(
        array_module.array(Z),
        coords={'Qx': (['Y', 'X'], Qx),
                'Qy': (['Y', 'X'], Qy)},
        dims=['Y', 'X'],
        name='Z',
    )

    x_coords = np.linspace(-100, 100, 3)
    y_coords = np.linspace(-200, 200, 5)
    out = xr.DataArray(array_module.array(
        [[nan, nan, nan],
         [nan, nan, nan],
         [nan, 6.,  nan],
         [nan, nan, nan],
         [nan, nan, nan]]
        ),
        coords=dict([
            ('Qx', x_coords),
            ('Qy', y_coords)]),
        dims=['Qy', 'Qx']
    )

    res = c.quadmesh(da, x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)

    res = c.quadmesh(da.transpose('X', 'Y', transpose_coords=True), x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)


@pytest.mark.parametrize(
    "bounds",
    (
        (-1, 1),  # malloc_consolidate(): unaligned fastbin chunk detected
        (-20023593.403653, 2991711.314653),  # double free or corruption (!prev)
    ),
)
def test_segfault_quadmesh(bounds):
    # Test for https://github.com/holoviz/datashader/issues/1431
    code = f"""\
    import datashader as dsh
    import numpy as np
    import xarray as xr

    left = -20037508.342789244
    bottom = -20037508.342789244
    right = 0.0
    top = 0.0
    cvs = dsh.Canvas(plot_height=512, plot_width=256, x_range=(left, right), y_range=(bottom, top))

    xsize = 10
    xmin, xmax = {bounds}

    y = np.array([
         97408.34081038, 69576.06157273, 41745.1070807 ,  13914.9473856,
        -13914.9473856, -41745.1070807, -69576.06157273, -97408.34081038,
        -125242.47486847, -153078.99399844,
    ])
    ysize = y.size

    da = xr.DataArray(
        np.ones((ysize, xsize)),
        dims=("y", "x"),
        coords={{"x": np.linspace(xmin, xmax, xsize), "y": y}},
        name="foo",
    )
    cvs.quadmesh(da, x="x", y="y")"""

    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)


@pytest.mark.parametrize('array_module', array_modules)
def test_rectilinear_quadmesh_bbox_smaller_than_grid(array_module):
    """Test for quadmesh with non-broadcast coordinates.

    This test addresses a bug where quadmesh returns all NaN values
    when coordinates are not properly broadcast for rectilinear grids.
    See: https://github.com/holoviz/datashader/issues/1438
    """

    west = 111_445.
    east = 111_483.
    south = 10_018_715.
    north = 10_018_754.

    da = xr.DataArray(
        np.array(
            [
                [-0.4246922, -0.41608012, -0.40739873],
                [-0.4381327, -0.42964128, -0.42107907],
                [-0.45110574, -0.4427344, -0.43429095],
            ],
            dtype=np.float32,
        ),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.array([9_000_000., 10_000_000., 11_000_000.]),
            "longitude": np.array([80_000., 111_000., 140_000.]),
        },
        name="foo",
    )

    # Canvas bbox (15-25, 150-250) overlaps with data coordinates (10-30, 100-300)
    cvs = ds.Canvas(64, 64, x_range=(west, east), y_range=(south, north))
    result = cvs.quadmesh(da, x="longitude", y="latitude")
    assert np.sum(np.isnan(result)) == 0

    result = cvs.quadmesh(da.isel(latitude=slice(None, None, -1)), x="longitude", y="latitude")
    assert np.sum(np.isnan(result)) == 0


@given(
    spacings=npst.arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=2, max_value=50),
            st.integers(min_value=2, max_value=50)
        ),
        elements=st.floats(
            min_value=0.1,  # Positive spacings to ensure monotonic coordinates
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False
        )
    ),
    start_value=st.floats(
        min_value=-1000,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False
    )
)
@settings(deadline=None)
def test_infer_interval_breaks_2d_consistency(spacings, start_value):
    """Test that infer_interval_breaks_2d matches sequential 1D application.

    This verifies that:
    infer_interval_breaks_2d(coords) ==
    infer_interval_breaks(infer_interval_breaks(coords, axis=1), axis=0)

    where coords are curvilinear coordinates constructed from cumulative sums.
    """
    # Construct curvilinear coordinates using cumsum of spacings
    # This ensures monotonic coordinates which is realistic for quadmesh data
    coords = np.cumsum(spacings, axis=0)
    coords = np.cumsum(coords, axis=1)
    coords = coords + start_value  # Add offset

    # Compute expected result using sequential 1D operations
    expected = infer_interval_breaks(infer_interval_breaks(coords, axis=1), axis=0)

    # Compute actual result using 2D operation
    actual = infer_interval_breaks_2d(coords)

    # Compare results
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('array_module', array_modules)
def test_raster_quadmesh_descending_coords(array_module):
    """
    Regression test for https://github.com/holoviz/datashader/issues/1439
    """
    west = 3125000.0
    south = 3250000.0
    east = 4250000.0
    north = 4375000.0

    # Create data with descending y coordinates (high to low)
    da = xr.DataArray(
        array_module.ones((940, 940)),
        dims=("x", "y"),
        coords={
            "x": np.linspace(3123580.0, 4250380.0, 940),
            "y": np.linspace(4376200.0, 3249400.0, 940),  # descending!
        },
        name="foo",
    )

    cvs = ds.Canvas(256, 256, x_range=(west, east), y_range=(south, north))
    result = cvs.quadmesh(da.transpose("y", "x"), x="x", y="y")
    assert result.isnull().sum().item() == 0

    result = cvs.quadmesh(da.isel(y=slice(None, None, -1)).transpose("y", "x"), x="x", y="y")
    assert result.isnull().sum().item() == 0


@pytest.mark.parametrize('array_module', array_modules)
def test_raster_quadmesh_descending_coords_2(array_module):
    """
    Regression test for https://github.com/holoviz/datashader/issues/1439
    """
    west=3125000.0
    south=4375000.0
    east=4250000.0
    north=5500000.0

    # Create data with descending y coordinates (high to low)
    da = xr.DataArray(
        array_module.ones((940, 868)),
        dims=("x", "y"),
        coords={
            "x": np.linspace(3123580.0, 4250380.0, 940),
            "y": np.linspace(5415400.0, 4375000.0, 868),  # descending!
        },
        name="foo",
    )

    cvs = ds.Canvas(256, 256, x_range=(west, east), y_range=(south, north))
    actual = cvs.quadmesh(da.transpose("y", "x"), x="x", y="y")
    expected = cvs.quadmesh(da.isel(y=slice(None, None, -1)).transpose("y", "x"), x="x", y="y")
    assert_eq_xr(expected, actual, close=True)


def test_rectilinear_extra_padding():
    from numpy import nan

    array = np.array([
        [        nan,         nan,         nan,         nan, -1.0571312 , -0.88049114, -0.6049668],
        [        nan,         nan,         nan,         nan, -1.2100513 , -1.0593421 , -0.7067303],
        [        nan,         nan,         nan,         nan,         nan, -1.4044759 , -1.3233978],
        [-1.2584106 ,         nan,         nan,         nan,         nan, -1.7786514 , -1.6885643],
        [-0.982517  , -1.0731102 ,         nan, -1.6481969 ,         nan,         nan, -1.9483067],
        [-0.7437196 , -0.86617017, -0.99102557, -1.4003996 , -1.6158952 , -2.17291   ,        nan],
        [-1.3059484 , -1.280411  , -1.3395019 , -1.5458245 , -1.7065538 , -1.954555  , -2.0651925]
    ], dtype=np.float32)
    x = np.array([-84, -79 , -73, -67, -62, -56, -51 ])
    y = np.array([5 , 13, 22, 30 , 39, 47, 56. ])
    da = xr.DataArray(array, dims=("y", "x"), coords={"x": x, "y": y}, name="foo")

    cvs = ds.Canvas(256, 256, x_range=(-72, -57), y_range=(26, 37))
    actual = cvs.quadmesh(da, x="x", y="y")
    assert actual.isel(x=1).isnull().all().item()
    assert actual.isel(x=0).isnull().all().item()

    # make sure canvas lines up with cell edges
    cvs = ds.Canvas(256, 256, x_range=(-70, -53.5), y_range=(17.5, 43))

    # insert nans along the border and a value in the center so the data
    # that is valid for the canvas is
    da.data[:, [3, 5]] = np.nan
    da.data[[2, 5], :] = np.nan
    da.data[3, 4] = 10
    expected_data = np.array([
           [np.nan, np.nan, np.nan],
           [np.nan,   10  , np.nan],
           [np.nan, np.nan, np.nan],
    ])
    np.testing.assert_array_equal(
        da.sel(x=slice(-70, -53.5), y=slice(17.5, 43)).data,
        expected_data,
    )

    actual = cvs.quadmesh(da, x="x", y="y")
    assert actual.isel(x=0).isnull().all().item()
    assert actual.isel(x=-1).isnull().all().item()
    assert actual.isel(y=0).isnull().all().item()
    assert actual.isel(y=-1).isnull().all().item()

    # make sure input data lines up with canvas
    actual_exact = cvs.quadmesh(da.isel(x=slice(3, 6), y=slice(2, 5)), x="x", y="y")
    assert_eq_xr(actual, actual_exact)

    actual_reversed = cvs.quadmesh(da.isel(x=slice(5, 2, -1), y=slice(4, 1, -1)), x="x", y="y")
    assert_eq_xr(actual, actual_reversed)


@pytest.mark.parametrize('xp', array_modules)
@pytest.mark.parametrize('size', [16, 64], ids=["upsample", "downsample"])
def test_quadmesh_3d_raster(rng, xp, size):
    cvs = ds.Canvas(
        plot_height=32, plot_width=32, x_range=(-1, 1), y_range=(-1, 1)
    )

    band = [0, 1, 2]
    data = xp.array(rng.random((size, size, len(band))))
    da = xr.DataArray(
        data,
        coords={
            "x": np.linspace(-1, 1, size),
            "y": np.linspace(-1, 1, size),
            "band": band,
        },
        dims=("y", "x", "band"),
        name="foo"
    )

    agg_3d = cvs.quadmesh(da.transpose(..., "y", "x"), x='x', y='y')
    for n in band:
        output = agg_3d.isel(band=n)
        expected = cvs.quadmesh(da.isel(band=n))
        expected = expected.assign_coords(band=n)
        assert_eq_xr(output, expected)

@pytest.mark.parametrize('xp', array_modules)
@pytest.mark.parametrize('size', [16, 64], ids=["upsample", "downsample"])
def test_quadmesh_3d_rectilinear(rng, xp, size):
    cvs = ds.Canvas(
        plot_height=32, plot_width=32, x_range=(-1, 1), y_range=(-1, 1)
    )

    band = [0, 1, 2]
    data = xp.array(rng.random((size, size, len(band))))

    # Create non-uniform coordinates to ensure rectilinear path
    x_coords = np.linspace(-1, 1, size)
    y_coords = np.linspace(-1, 1, size)
    # Add small random perturbations to break uniformity
    x_coords = x_coords + rng.uniform(-0.001, 0.001, size)
    y_coords = y_coords + rng.uniform(-0.001, 0.001, size)

    da = xr.DataArray(
        data,
        coords={
            "x": x_coords,
            "y": y_coords,
            "band": band,
        },
        dims=("y", "x", "band"),
        name="foo"
    )

    agg_3d = cvs.quadmesh(da.transpose(..., "y", "x"), x='x', y='y')
    for n in band:
        output = agg_3d.isel(band=n)
        expected = cvs.quadmesh(da.isel(band=n))
        expected = expected.assign_coords(band=n)
        assert_eq_xr(output, expected)


@pytest.mark.parametrize('xp', array_modules)
@pytest.mark.parametrize('size', [16, 64], ids=["upsample", "downsample"])
def test_quadmesh_3d_curvilinear(rng, xp, size):
    cvs = ds.Canvas(
        plot_height=32, plot_width=32, x_range=(-1, 1), y_range=(-1, 1)
    )

    band = [0, 1, 2]
    data = xp.array(rng.random((size, size, len(band))))

    # Create 2D coordinate arrays (curvilinear)
    x_1d = xp.linspace(-1, 1, size)
    y_1d = xp.linspace(-1, 1, size)
    x_2d, y_2d = xp.meshgrid(x_1d, y_1d, indexing='xy')

    da = xr.DataArray(
        data,
        coords={
            "x": (["y", "x"], x_2d),
            "y": (["y", "x"], y_2d),
            "band": band,
        },
        dims=("y", "x", "band"),
        name="foo"
    )

    agg_3d = cvs.quadmesh(da.transpose(..., "y", "x"), x='x', y='y')
    for n in band:
        output = agg_3d.isel(band=n)
        expected = cvs.quadmesh(da.isel(band=n), x='x', y='y')
        expected = expected.assign_coords(band=n)
        assert_eq_xr(output, expected)
