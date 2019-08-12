import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds


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
    print(res)
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
