from __future__ import annotations
import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
import pytest

import dask.array
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr

array_modules = [np, dask.array]
try:
    import cudf
    import cupy
    array_modules.append(cupy)
except ImportError:
    cudf = None
    cupy = None


dask.config.set(scheduler='single-threaded')


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
