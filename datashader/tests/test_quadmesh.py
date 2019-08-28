from __future__ import absolute_import
import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
from collections import OrderedDict


def test_rect_quadmesh_autorange():
    c = ds.Canvas(plot_width=8, plot_height=4)
    da = xr.DataArray(
        [[1, 2, 3, 4],
         [5, 6, 7, 8]],
        coords=[('b', [1, 2]),
                ('a', [1, 2, 3, 4])],
        name='Z')

    y_coords = np.linspace(0.75, 2.25, 4)
    x_coords = np.linspace(0.75, 4.25, 8)
    out = xr.DataArray(np.array(
        [[1., 1., 2., 2., 3., 3., 4., 4.],
         [1., 1., 2., 2., 3., 3., 4., 4.],
         [5., 5., 6., 6., 7., 7., 8., 8.],
         [5., 5., 6., 6., 7., 7., 8., 8.]],
        dtype='i4'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert res.equals(out)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert res.equals(out)


def test_rect_quadmesh_autorange_reversed():
    c = ds.Canvas(plot_width=8, plot_height=4)
    da = xr.DataArray(
        [[1, 2, 3, 4],
         [5, 6, 7, 8]],
        coords=[('b', [-1, -2]),
                ('a', [-1, -2, -3, -4])],
        name='Z')

    y_coords = np.linspace(-2.25, -0.75, 4)
    x_coords = np.linspace(-4.25, -0.75, 8)
    out = xr.DataArray(np.array(
        [[8., 8., 7., 7., 6., 6., 5., 5.],
         [8., 8., 7., 7., 6., 6., 5., 5.],
         [4., 4., 3., 3., 2., 2., 1., 1.],
         [4., 4., 3., 3., 2., 2., 1., 1.]],
        dtype='i4'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert res.equals(out)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert res.equals(out)


def test_rect_quadmesh_manual_range():
    c = ds.Canvas(plot_width=8, plot_height=4,
                  x_range=[1, 3],
                  y_range=[-1, 3])

    da = xr.DataArray(
        [[1, 2, 3, 4],
         [5, 6, 7, 8]],
        coords=[('b', [1, 2]),
                ('a', [1, 2, 3, 4])],
        name='Z')

    y_coords = np.linspace(-0.5, 2.5, 4)
    x_coords = np.linspace(1.125, 2.875, 8)
    out = xr.DataArray(np.array(
        [[nan, nan, nan, nan, nan, nan, nan, nan],
         [1., 1., 2., 2., 2., 2., 3., 3.],
         [5., 5., 6., 6., 6., 6., 7., 7.],
         [nan, nan, nan, nan, nan, nan, nan, nan]],
        dtype='f4'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert res.equals(out)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert res.equals(out)


def test_subpixel_quads_represented():
    c = ds.Canvas(plot_width=8, plot_height=4,
                  x_range=[0, 16],
                  y_range=[0, 8])

    da = xr.DataArray(
        [[1, 2, 3, 4],
         [5, 6, 7, 8]],
        coords=[('b', [1, 2]),
                ('a', [1, 2, 3, 4])],
        name='Z')

    y_coords = np.linspace(1, 7, 4)
    x_coords = np.linspace(1, 15, 8)
    out = xr.DataArray(np.array(
        [[14., 22., nan, nan, nan, nan, nan, nan],
         [nan, nan, nan, nan, nan, nan, nan, nan],
         [nan, nan, nan, nan, nan, nan, nan, nan],
         [nan, nan, nan, nan, nan, nan, nan, nan]],
        dtype='f4'),
        coords=[('b', y_coords),
                ('a', x_coords)]
    )

    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert res.equals(out)

    # Check transpose gives same answer
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert res.equals(out)


def test_curve_quadmesh_rect_autorange():
    c = ds.Canvas(plot_width=8, plot_height=4)
    Qx = np.array(
        [[1, 2],
         [1, 2]]
    )
    Qy = np.array(
        [[1, 1],
         [2, 2]]
    )
    Z = np.arange(4, dtype='int32').reshape(2, 2)
    da = xr.DataArray(
        Z,
        coords={'Qx': (['Y', 'X'], Qx),
                'Qy': (['Y', 'X'], Qy)},
        dims=['Y', 'X'],
        name='Z',
    )

    y_coords = np.linspace(0.75, 2.25, 4)
    x_coords = np.linspace(0.625, 2.375, 8)
    out = xr.DataArray(np.array(
        [[0., 0., 0., 0., 1., 1., 1., 1.],
         [0., 0., 0., 0., 1., 1., 1., 1.],
         [2., 2., 2., 2., 3., 3., 3., 3.],
         [2., 2., 2., 2., 3., 3., 3., 3.]],
        ),
        coords=[('Qy', y_coords),
                ('Qx', x_coords)]
    )

    res = c.quadmesh(da, x='Qx', y='Qy', agg=ds.sum('Z'))
    assert res.equals(out)

    res = c.quadmesh(da.transpose('X', 'Y'), x='Qx', y='Qy', agg=ds.sum('Z'))
    assert res.equals(out)


def test_curve_quadmesh_autorange():
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
        Z,
        coords={'Qx': (['Y', 'X'], Qx),
                'Qy': (['Y', 'X'], Qy)},
        dims=['Y', 'X'],
        name='Z',
    )

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
        coords=OrderedDict([
            ('Qx', x_coords),
            ('Qy', y_coords)]),
        dims=['Qy', 'Qx']
    )

    res = c.quadmesh(da, x='Qx', y='Qy', agg=ds.sum('Z'))
    assert res.equals(out)

    res = c.quadmesh(da.transpose('X', 'Y'), x='Qx', y='Qy', agg=ds.sum('Z'))
    assert res.equals(out)


def test_curve_quadmesh_manual_range():
    c = ds.Canvas(plot_width=4, plot_height=8, x_range=[1, 2], y_range=[1, 3])
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
        Z,
        coords={'Qx': (['Y', 'X'], Qx),
                'Qy': (['Y', 'X'], Qy)},
        dims=['Y', 'X'],
        name='Z',
    )

    x_coords = np.linspace(1.125, 1.875, 4)
    y_coords = np.linspace(1.125, 2.875, 8)
    out = xr.DataArray(np.array(
        [[0., 0., 1., 1.],
         [0., 0., 1., 1.],
         [0., 0., 1., 1.],
         [0., 0., 1., 3.],
         [0., 0., 3., 3.],
         [0., 2., 3., 3.],
         [2., 2., 3., 3.],
         [2., 2., 3., 3.]]
        ),
        coords=OrderedDict([
            ('Qx', x_coords),
            ('Qy', y_coords)]),
        dims=['Qy', 'Qx']
    )

    res = c.quadmesh(da, x='Qx', y='Qy', agg=ds.sum('Z'))
    assert res.equals(out)

    res = c.quadmesh(da.transpose('X', 'Y'), x='Qx', y='Qy', agg=ds.sum('Z'))
    assert res.equals(out)


def test_curve_quadmesh_manual_range_subpixel():
    c = ds.Canvas(plot_width=3, plot_height=5,
                  x_range=[-150, 150], y_range=[-250, 250])
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
        Z,
        coords={'Qx': (['Y', 'X'], Qx),
                'Qy': (['Y', 'X'], Qy)},
        dims=['Y', 'X'],
        name='Z',
    )

    x_coords = np.linspace(-100, 100, 3)
    y_coords = np.linspace(-200, 200, 5)
    out = xr.DataArray(np.array(
        [[nan, nan, nan],
         [nan, nan, nan],
         [nan, 6.,  nan],
         [nan, nan, nan],
         [nan, nan, nan]]
        ),
        coords=OrderedDict([
            ('Qx', x_coords),
            ('Qy', y_coords)]),
        dims=['Qy', 'Qx']
    )

    res = c.quadmesh(da, x='Qx', y='Qy', agg=ds.sum('Z'))
    assert res.equals(out)

    res = c.quadmesh(da.transpose('X', 'Y'), x='Qx', y='Qy', agg=ds.sum('Z'))
    assert res.equals(out)
