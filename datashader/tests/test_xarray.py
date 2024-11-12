from __future__ import annotations
from copy import deepcopy
import numpy as np
from numpy import nan
import xarray as xr

import datashader as ds
from datashader.tests.test_pandas import assert_eq_ndarray

import pytest

try:
    import cupy
except ImportError:
    cupy = None

xda = xr.DataArray(data=np.array([1.] * 10 + [10] * 10),
                   dims=('record'),
                   coords={'x': xr.DataArray(np.array([0.]*10 + [1]*10), dims=('record')),
                           'y': xr.DataArray(np.array([0.]*5 + [1]*5 + [0]*5 + [1]*5),
                                             dims=('record')),
                           'i32': xr.DataArray(np.arange(20, dtype='i4'), dims=('record')),
                           'i64': xr.DataArray(np.arange(20, dtype='i8'), dims=('record')),
                           'f32': xr.DataArray(np.arange(20, dtype='f4'), dims=('record')),
                           'f64': xr.DataArray(np.arange(20, dtype='f8'), dims=('record')),
                   })
xda.f32[2] = np.nan
xda.f64[2] = np.nan
xds = xda.to_dataset(name='value').reset_coords(names=['i32', 'i64'])

try:
    import dask
    xdda = xda.chunk(chunks=5)
    xdds = xds.chunk(chunks=5)
except ImportError:
    dask, xdda, xdds = None, None, None

c = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))

axis = ds.core.LinearAxis()
lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)
coords = [lincoords, lincoords]
dims = ['y', 'x']


def assert_eq(agg, b):
    assert agg.equals(b)


@pytest.mark.parametrize("source", [xda, xdda, xds, xdds])
def test_count(source):
    if source is None:
        pytest.skip("Dask not available")
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


x = np.arange(5)
channel = np.arange(2)
value = [-33, -55]
other = [2.2, 1.1]
data = np.array([[2, 1, 0, 1, 2], [1, 1, 1, 1, 1]], dtype=np.float64)
ds2d_x0 = xr.Dataset(
    data_vars=dict(
        name=(("x", "channel"), data.T.copy()),
        value=("channel", value),
        other=("channel", other),
    ),
    coords=dict(
        channel=("channel", channel),
        x=("x", x),
    ),
)
ds2d_x1 = xr.Dataset(
    data_vars=dict(
        name=(("channel", "x"), data),
        value=("channel", value),
        other=("channel", other),
    ),
    coords=dict(
        channel=("channel", channel),
        x=("x", x),
    ),
)
ds2ds = [ds2d_x0, ds2d_x1]


@pytest.mark.parametrize("ds2d", ds2ds)
@pytest.mark.parametrize('on_gpu', [False, pytest.param(True, marks=pytest.mark.gpu)])
@pytest.mark.parametrize("chunksizes", [
    None,
    dict(x=10, channel=10),
    dict(x=10, channel=1),
    dict(x=3, channel=10),
    dict(x=3, channel=1),
])
def test_lines_xarray_common_x(ds2d, on_gpu, chunksizes):
    source = deepcopy(ds2d)
    if on_gpu:
        if chunksizes is not None:
            pytest.skip("CUDA-dask for LinesXarrayCommonX not implemented")

        # CPU -> GPU
        source.name.data = cupy.asarray(source.name.data)

    if chunksizes is not None:
        if dask is None:
            pytest.skip("Dask not available")
        source = source.chunk(chunksizes)

    canvas = ds.Canvas(plot_height=3, plot_width=7)

    # Expected solutions
    sol_count = np.array(
        [[0, 0, 1, 1, 0, 0, 0], [1, 2, 1, 1, 2, 2, 1], [1, 0, 0, 0, 0, 0, 1]],
        dtype=np.uint32)
    sol_max = np.array(
        [[nan, nan, -33, -33, nan, nan, nan], [-55, -33, -55, -55, -33, -33, -55], [-33, nan, nan, nan, nan, nan, -33]],  # noqa: E501
        dtype=np.float64)
    sol_min = np.array(
        [[nan, nan, -33, -33, nan, nan, nan], [-55, -55, -55, -55, -55, -55, -55], [-33, nan, nan, nan, nan, nan, -33]],  # noqa: E501
        dtype=np.float64)
    sol_sum = np.array(
        [[nan, nan, -33, -33, nan, nan, nan], [-55, -88, -55, -55, -88, -88, -55], [-33, nan, nan, nan, nan, nan, -33]],  # noqa: E501
        dtype=np.float64)
    sol_max_row_index = np.array(
        [[-1, -1, 0, 0, -1, -1, -1], [1, 1, 1, 1, 1, 1, 1], [0, -1, -1, -1, -1, -1, 0]],
        dtype=np.int64)
    sol_min_row_index = np.array(
        [[-1, -1, 0, 0, -1, -1, -1], [1, 0, 1, 1, 0, 0, 1], [0, -1, -1, -1, -1, -1, 0]],
        dtype=np.int64)

    if chunksizes is not None and chunksizes["x"] == 3:
        # Dask chunking in x-direction gives different (incorrect) results.
        sol_count[:, 4] = 0
        sol_max[:, 4] = nan
        sol_min[:, 4] = nan
        sol_sum[:, 4] = nan
        sol_max_row_index[:, 4] = -1
        sol_min_row_index[:, 4] = -1

    sol_first = np.select([sol_min_row_index==0, sol_min_row_index==1], value, np.nan)
    sol_last = np.select([sol_max_row_index==0, sol_max_row_index==1], value, np.nan)
    sol_where_max_other = np.select([sol_max==-33, sol_max==-55], other, np.nan)
    sol_where_max_row = np.select([sol_max==-33, sol_max==-55], [0, 1], -1)
    sol_where_min_other = np.select([sol_min==-33, sol_min==-55], other, np.nan)
    sol_where_min_row = np.select([sol_min==-33, sol_min==-55], [0, 1], -1)

    # count
    agg = canvas.line(source, x="x", y="name", agg=ds.count())
    assert_eq_ndarray(agg.x_range, (0, 4), close=True)
    assert_eq_ndarray(agg.y_range, (0, 2), close=True)
    assert_eq_ndarray(agg.data, sol_count)
    assert isinstance(agg.data, cupy.ndarray if on_gpu else np.ndarray)

    # any
    agg = canvas.line(source, x="x", y="name", agg=ds.any())
    assert_eq_ndarray(agg.data, sol_count > 0)

    # max
    agg = canvas.line(source, x="x", y="name", agg=ds.max("value"))
    assert_eq_ndarray(agg.data, sol_max)

    # min
    agg = canvas.line(source, x="x", y="name", agg=ds.min("value"))
    assert_eq_ndarray(agg.data, sol_min)

    # sum
    agg = canvas.line(source, x="x", y="name", agg=ds.sum("value"))
    assert_eq_ndarray(agg.data, sol_sum)

    # _max_row_index
    agg = canvas.line(source, x="x", y="name", agg=ds._max_row_index())
    assert_eq_ndarray(agg.data, sol_max_row_index)

    # _min_row_index
    agg = canvas.line(source, x="x", y="name", agg=ds._min_row_index())
    assert_eq_ndarray(agg.data, sol_min_row_index)

    # first
    agg = canvas.line(source, x="x", y="name", agg=ds.first("value"))
    assert_eq_ndarray(agg.data, sol_first)

    # last
    agg = canvas.line(source, x="x", y="name", agg=ds.last("value"))
    assert_eq_ndarray(agg.data, sol_last)

    # where(max) returning other row
    agg = canvas.line(source, x="x", y="name", agg=ds.where(ds.max("value"), "other"))
    assert_eq_ndarray(agg.data, sol_where_max_other)

    # where(max) returning row index
    agg = canvas.line(source, x="x", y="name", agg=ds.where(ds.max("value")))
    assert_eq_ndarray(agg.data, sol_where_max_row)

    # where(min) returning other row
    agg = canvas.line(source, x="x", y="name", agg=ds.where(ds.min("value"), "other"))
    assert_eq_ndarray(agg.data, sol_where_min_other)

    # where(min) returning row index
    agg = canvas.line(source, x="x", y="name", agg=ds.where(ds.min("value")))
    assert_eq_ndarray(agg.data, sol_where_min_row)
