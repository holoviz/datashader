from __future__ import annotations
import numpy as np
import xarray as xr

import datashader as ds

import pytest


xda = xr.DataArray(data=np.array(([1.] * 10 + [10] * 10)),
                   dims=('record'),
                   coords={'x': xr.DataArray(np.array(([0.] * 10 + [1] * 10)), dims=('record')),
                           'y': xr.DataArray(np.array(([0.] * 5 + [1] * 5 + [0] * 5 + [1] * 5)), dims=('record')),
                           'i32': xr.DataArray(np.arange(20, dtype='i4'), dims=('record')),
                           'i64': xr.DataArray(np.arange(20, dtype='i8'), dims=('record')),
                           'f32': xr.DataArray(np.arange(20, dtype='f4'), dims=('record')),
                           'f64': xr.DataArray(np.arange(20, dtype='f8'), dims=('record')),
                   })
xda.f32[2] = np.nan
xda.f64[2] = np.nan
xds = xda.to_dataset(name='value').reset_coords(names=['i32', 'i64'])

xdda = xda.chunk(chunks=5)
xdds = xds.chunk(chunks=5)

c = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))

axis = ds.core.LinearAxis()
lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)
coords = [lincoords, lincoords]
dims = ['y', 'x']


def assert_eq(agg, b):
    assert agg.equals(b)


@pytest.mark.parametrize("source", [
    (xda), (xdda), (xds), (xdds),
])
def test_count(source):
    out = xr.DataArray(np.array([[5, 5], [5, 5]], dtype='i4'),
                       coords=coords, dims=dims)
    assert_eq(c.points(source, 'x', 'y', ds.count('i32')), out)
    assert_eq(c.points(source, 'x', 'y', ds.count('i64')), out)
    assert_eq(c.points(source, 'x', 'y', ds.count()), out)
    assert_eq(c.points(source, 'x', 'y', ds.count('value')), out)
    out = xr.DataArray(np.array([[4, 5], [5, 5]], dtype='i4'),
                       coords=coords, dims=dims)
    assert_eq(c.points(source, 'x', 'y', ds.count('f32')), out)

    agg = c.points(source, 'x', 'y', ds.count('f64'))
    assert_eq(agg, out)
    np.testing.assert_array_almost_equal(agg.x_range, (0, 1))
    np.testing.assert_array_almost_equal(agg.y_range, (0, 1))

