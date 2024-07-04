from __future__ import annotations
import pytest

from os import path
from itertools import product

import datashader as ds
import xarray as xr
import numpy as np
import pandas as pd

from datashader.resampling import compute_chunksize
import datashader.transfer_functions as tf
from packaging.version import Version
from .utils import dask_skip

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import rioxarray
except ImportError:
    rioxarray = None

try:
    from dask.context import config
    import dask.array as da
    config.set(scheduler='synchronous')
except ImportError:
    da = None

open_rasterio_available = pytest.mark.skipif(rioxarray is None and rasterio is None,
                                             reason="requires rioxarray or rasterio")

BASE_PATH = path.split(__file__)[0]
DATA_PATH = path.abspath(path.join(BASE_PATH, 'data'))
TEST_RASTER_PATH = path.join(DATA_PATH, 'world.rgb.tif')


def open_rasterio(path, *args, **kwargs):
    # xarray deprecated xr.open_rasterio in its 0.20 release
    # in favor or rioxarray.open_rasterio.
    if Version(xr.__version__) < Version('0.20'):
        func = xr.open_rasterio
    else:
        func = rioxarray.open_rasterio
    return func(path, *args, **kwargs)


@pytest.fixture
def cvs():
    with open_rasterio(TEST_RASTER_PATH) as src:
        res = ds.utils.calc_res(src)
        left, bottom, right, top = ds.utils.calc_bbox(src.x.values, src.y.values, res)
        return ds.Canvas(plot_width=2,
                         plot_height=2,
                         x_range=(left, right),
                         y_range=(bottom, top))


@open_rasterio_available
def test_raster_aggregate_default(cvs):
    with open_rasterio(TEST_RASTER_PATH) as src:
        agg = cvs.raster(src)
        assert agg is not None

@open_rasterio_available
def test_raster_aggregate_nearest(cvs):
    with open_rasterio(TEST_RASTER_PATH) as src:
        agg = cvs.raster(src, upsample_method='nearest')
        assert agg is not None


@pytest.mark.skip('use_overviews opt no longer supported; may be re-implemented in the future')
@open_rasterio_available
def test_raster_aggregate_with_overviews(cvs):
    with open_rasterio(TEST_RASTER_PATH) as src:
        agg = cvs.raster(src, use_overviews=True)
        assert agg is not None


@pytest.mark.skip('use_overviews opt no longer supported; may be re-implemented in the future')
@open_rasterio_available
def test_raster_aggregate_without_overviews(cvs):
    with open_rasterio(TEST_RASTER_PATH) as src:
        agg = cvs.raster(src, use_overviews=False)
        assert agg is not None


@open_rasterio_available
def test_out_of_bounds_return_correct_size(cvs):
    with open_rasterio(TEST_RASTER_PATH) as src:
        cvs = ds.Canvas(plot_width=2,
                        plot_height=2,
                        x_range=[1e10, 1e20],
                        y_range=[1e10, 1e20])
        try:
            cvs.raster(src)
        except ValueError:
            pass
        else:
            assert False


@open_rasterio_available
def test_partial_extent_returns_correct_size():
    with open_rasterio(TEST_RASTER_PATH) as src:
        res = ds.utils.calc_res(src)
        left, bottom, right, top = ds.utils.calc_bbox(src.x.values, src.y.values, res)
        half_width = (right - left) / 2
        half_height = (top - bottom) / 2
        cvs = ds.Canvas(plot_width=512,
                        plot_height=256,
                        x_range=[left-half_width, left+half_width],
                        y_range=[bottom-half_height, bottom+half_height])
        agg = cvs.raster(src)
        assert agg.shape == (3, 256, 512)
        assert agg is not None


@open_rasterio_available
def test_partial_extent_with_layer_returns_correct_size(cvs):
    with open_rasterio(TEST_RASTER_PATH) as src:
        res = ds.utils.calc_res(src)
        left, bottom, right, top = ds.utils.calc_bbox(src.x.values, src.y.values, res)
        half_width = (right - left) / 2
        half_height = (top - bottom) / 2
        cvs = ds.Canvas(plot_width=512,
                        plot_height=256,
                        x_range=[left-half_width, left+half_width],
                        y_range=[bottom-half_height, bottom+half_height])
        agg = cvs.raster(src, layer=1)
        assert agg.shape == (256, 512)
        assert agg is not None


@open_rasterio_available
def test_full_extent_returns_correct_coords():
    with open_rasterio(TEST_RASTER_PATH) as src:
        res = ds.utils.calc_res(src)
        left, bottom, right, top = ds.utils.calc_bbox(src.x.values, src.y.values, res)
        cvs = ds.Canvas(plot_width=512,
                        plot_height=256,
                        x_range=[left, right],
                        y_range=[bottom, top])
        agg = cvs.raster(src)
        assert agg.shape == (3, 256, 512)
        assert agg is not None
        for dim in src.dims:
            assert np.all(agg[dim].data == src[dim].data)

        assert np.allclose(agg.x_range, (-180, 180))
        assert np.allclose(agg.y_range, (-90, 90))


@open_rasterio_available
def test_calc_res():
    """Assert that resolution is calculated correctly when using the xarray
    rasterio backend.
    """
    import rasterio

    with open_rasterio(TEST_RASTER_PATH) as src:
        xr_res = ds.utils.calc_res(src)
    with rasterio.open(TEST_RASTER_PATH) as src:
        rio_res = src.res
    assert np.allclose(xr_res, rio_res)


@open_rasterio_available
def test_calc_bbox():
    """Assert that bounding boxes are calculated correctly when using the xarray
    rasterio backend.
    """
    import rasterio

    with open_rasterio(TEST_RASTER_PATH) as src:
        xr_res = ds.utils.calc_res(src)
        xr_bounds = ds.utils.calc_bbox(src.x.values, src.y.values, xr_res)
    with rasterio.open(TEST_RASTER_PATH) as src:
        rio_bounds = src.bounds
    assert np.allclose(xr_bounds, rio_bounds, atol=1.0)  # allow for absolute diff of 1.0


def test_raster_both_ascending():
    """
    Assert raster with ascending x- and y-coordinates is aggregated correctly.
    """
    xs = np.arange(10)
    ys = np.arange(5)
    arr = xs*ys[np.newaxis].T
    xarr = xr.DataArray(arr, coords={'X': xs, 'Y': ys}, dims=['Y', 'X'])
    cvs = ds.Canvas(10, 5, x_range=(-.5, 9.5), y_range=(-.5, 4.5))
    agg = cvs.raster(xarr)

    assert np.allclose(agg.data, arr)
    assert np.allclose(agg.X.values, xs)
    assert np.allclose(agg.Y.values, ys)
    assert np.allclose(agg.x_range, (-0.5, 9.5))
    assert np.allclose(agg.y_range, (-0.5, 4.5))


def test_raster_both_ascending_partial_range():
    """
    Assert raster with ascending x- and y-coordinates and a partial canvas
    range is aggregated correctly.
    """
    xs = np.arange(10)
    ys = np.arange(5)
    arr = xs*ys[np.newaxis].T
    xarr = xr.DataArray(arr, coords={'X': xs, 'Y': ys}, dims=['Y', 'X'])
    cvs = ds.Canvas(7, 3, x_range=(.5, 7.5), y_range=(.5, 3.5))
    agg = cvs.raster(xarr)

    assert np.allclose(agg.data, xarr.sel(X=slice(1, 7), Y=slice(1, 3)))
    assert np.allclose(agg.X.values, xs[1:8])
    assert np.allclose(agg.Y.values, ys[1:4])
    assert np.allclose(agg.x_range, (0.5, 7.5))
    assert np.allclose(agg.y_range, (0.5, 3.5))


def test_raster_both_descending():
    """
    Assert raster with ascending x- and y-coordinates is aggregated correctly.
    """
    xs = np.arange(10)[::-1]
    ys = np.arange(5)[::-1]
    arr = xs*ys[np.newaxis].T
    xarr = xr.DataArray(arr, coords={'X': xs, 'Y': ys}, dims=['Y', 'X'])
    cvs = ds.Canvas(10, 5, x_range=(-.5, 9.5), y_range=(-.5, 4.5))
    agg = cvs.raster(xarr)

    assert np.allclose(agg.data, arr)
    assert np.allclose(agg.X.values, xs)
    assert np.allclose(agg.Y.values, ys)
    assert np.allclose(agg.x_range, (-0.5, 9.5))
    assert np.allclose(agg.y_range, (-0.5, 4.5))


def test_raster_both_descending_partial_range():
    """
    Assert raster with ascending x- and y-coordinates and a partial canvas range
    is aggregated correctly.
    """
    xs = np.arange(10)[::-1]
    ys = np.arange(5)[::-1]
    arr = xs*ys[np.newaxis].T
    xarr = xr.DataArray(arr, coords={'X': xs, 'Y': ys}, dims=['Y', 'X'])
    cvs = ds.Canvas(7, 3, x_range=(.5, 7.5), y_range=(.5, 3.5))
    agg = cvs.raster(xarr)

    assert np.allclose(agg.data, xarr.sel(Y=slice(3,1), X=slice(7, 1)).data)
    assert np.allclose(agg.X.values, xs[2:9])
    assert np.allclose(agg.Y.values, ys[1:4])
    assert np.allclose(agg.x_range, (0.5, 7.5))
    assert np.allclose(agg.y_range, (0.5, 3.5))


def test_raster_x_ascending_y_descending():
    """
    Assert raster with ascending x- and descending y-coordinates is aggregated correctly.
    """
    xs = np.arange(10)
    ys = np.arange(5)[::-1]
    arr = xs*ys[np.newaxis].T
    xarr = xr.DataArray(arr, coords={'X': xs, 'Y': ys}, dims=['Y', 'X'])
    cvs = ds.Canvas(10, 5, x_range=(-.5, 9.5), y_range=(-.5, 4.5))
    agg = cvs.raster(xarr)

    assert np.allclose(agg.data, arr)
    assert np.allclose(agg.X.values, xs)
    assert np.allclose(agg.Y.values, ys)
    assert np.allclose(agg.x_range, (-0.5, 9.5))
    assert np.allclose(agg.y_range, (-0.5, 4.5))


def test_raster_x_ascending_y_descending_partial_range():
    """
    Assert raster with ascending x- and descending y-coordinates is aggregated correctly.
    """
    xs = np.arange(10)
    ys = np.arange(5)[::-1]
    arr = xs*ys[np.newaxis].T
    xarr = xr.DataArray(arr, coords={'X': xs, 'Y': ys}, dims=['Y', 'X'])
    cvs = ds.Canvas(7, 2, x_range=(0.5, 7.5), y_range=(1.5, 3.5))
    agg = cvs.raster(xarr)

    assert np.allclose(agg.data, xarr.sel(X=slice(1, 7), Y=slice(3, 2)).data)
    assert np.allclose(agg.X.values, xs[1:8])
    assert np.allclose(agg.Y.values, ys[1:3])
    assert np.allclose(agg.x_range, (0.5, 7.5))
    assert np.allclose(agg.y_range, (1.5, 3.5))


def test_raster_x_descending_y_ascending():
    """
    Assert raster with descending x- and ascending y-coordinates is aggregated correctly.
    """
    xs = np.arange(10)[::-1]
    ys = np.arange(5)
    arr = xs*ys[np.newaxis].T
    xarr = xr.DataArray(arr, coords={'X': xs, 'Y': ys}, dims=['Y', 'X'])
    cvs = ds.Canvas(10, 5, x_range=(-.5, 9.5), y_range=(-.5, 4.5))
    agg = cvs.raster(xarr)

    assert np.allclose(agg.data, arr)
    assert np.allclose(agg.X.values, xs)
    assert np.allclose(agg.Y.values, ys)
    assert np.allclose(agg.x_range, (-0.5, 9.5))
    assert np.allclose(agg.y_range, (-0.5, 4.5))


def test_raster_x_descending_y_ascending_partial_range():
    """
    Assert raster with descending x- and ascending y-coordinates is aggregated correctly.
    """
    xs = np.arange(10)[::-1]
    ys = np.arange(5)
    arr = xs*ys[np.newaxis].T
    xarr = xr.DataArray(arr, coords={'X': xs, 'Y': ys}, dims=['Y', 'X'])
    cvs = ds.Canvas(7, 2, x_range=(.5, 7.5), y_range=(1.5, 3.5))
    agg = cvs.raster(xarr)

    assert np.allclose(agg.data, xarr.sel(X=slice(7, 1), Y=slice(2, 3)).data)
    assert np.allclose(agg.X.values, xs[2:9])
    assert np.allclose(agg.Y.values, ys[2:4])


def test_raster_integer_nan_value():
    """
    Ensure custom nan_value is handled correctly for integer arrays.
    """
    cvs = ds.Canvas(plot_height=2, plot_width=2, x_range=(0, 1), y_range=(0,1))
    array = np.array([[9999, 1, 2, 3], [4, 9999, 6, 7], [8, 9, 9999, 11]])
    coords = {'x': np.linspace(0, 1, 4), 'y': np.linspace(0, 1, 3)}
    xr_array = xr.DataArray(array, coords=coords, dims=['y', 'x'])

    agg = cvs.raster(xr_array, downsample_method='max', nan_value=9999)
    expected = np.array([[4, 7], [9, 11]])

    assert np.allclose(agg.data, expected)
    assert agg.data.dtype.kind == 'i'
    assert np.allclose(agg.x.values, np.array([0.25, 0.75]))
    assert np.allclose(agg.y.values, np.array([0.25, 0.75]))


def test_raster_float_nan_value():
    """
    Ensure default nan_value is handled correctly for float arrays
    """
    cvs = ds.Canvas(plot_height=2, plot_width=2, x_range=(0, 1), y_range=(0,1))
    array = np.array([[np.nan, 1., 2., 3.], [4., np.nan, 6., 7.], [8., 9., np.nan, 11.]])
    coords = {'x': np.linspace(0, 1, 4), 'y': np.linspace(0, 1, 3)}
    xr_array = xr.DataArray(array, coords=coords, dims=['y', 'x'])

    agg = cvs.raster(xr_array, downsample_method='max')
    expected = np.array([[4, 7], [9, 11]])

    assert np.allclose(agg.data, expected)
    assert agg.data.dtype.kind == 'f'
    assert np.allclose(agg.x.values, np.array([0.25, 0.75]))
    assert np.allclose(agg.y.values, np.array([0.25, 0.75]))


def test_raster_integer_nan_value_padding():
    """
    Ensure that the padding values respect the supplied nan_value.
    """

    cvs = ds.Canvas(plot_height=3, plot_width=3, x_range=(0, 2), y_range=(0, 2))
    array = np.array([[9999, 1, 2, 3], [4, 9999, 6, 7], [8, 9, 9999, 11]])
    xr_array = xr.DataArray(array, coords={'x': np.linspace(0, 1, 4), 'y': np.linspace(0, 1, 3)},
                            dims=['y', 'x'])

    agg = cvs.raster(xr_array, downsample_method='max', nan_value=9999)
    expected = np.array([[4, 7, 9999], [9, 11, 9999], [9999, 9999, 9999]])

    assert np.allclose(agg.data, expected)
    assert agg.data.dtype.kind == 'i'
    assert np.allclose(agg.x.values, np.array([1/3., 1.0, 5/3.]))
    assert np.allclose(agg.y.values, np.array([1/3., 1.0, 5/3.]))


def test_raster_float_nan_value_padding():
    """
    Ensure that the padding values respect the supplied nan_value.
    """

    cvs = ds.Canvas(plot_height=3, plot_width=3, x_range=(0, 2), y_range=(0, 2))
    array = np.array([[np.nan, 1., 2., 3.], [4., np.nan, 6., 7.], [8., 9., np.nan, 11.]])
    xr_array = xr.DataArray(array, coords={'x': np.linspace(0, 1, 4), 'y': np.linspace(0, 1, 3)},
                            dims=['y', 'x'])

    agg = cvs.raster(xr_array, downsample_method='max')
    expected = np.array([[4., 7., np.nan], [9., 11., np.nan], [np.nan, np.nan, np.nan]])

    assert np.allclose(agg.data, expected, equal_nan=True)
    assert agg.data.dtype.kind == 'f'
    assert np.allclose(agg.x.values, np.array([1/3., 1.0, 5/3.]))
    assert np.allclose(agg.y.values, np.array([1/3., 1.0, 5/3.]))


def test_raster_single_pixel_range():
    """
    Ensure that canvas range covering a single pixel are handled correctly.
    """

    cvs = ds.Canvas(plot_height=3, plot_width=3, x_range=(0, 0.1), y_range=(0, 0.1))
    array = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    xr_array = xr.DataArray(array, dims=['y', 'x'],
                            coords={'x': np.linspace(0, 1, 4),
                                    'y': np.linspace(0, 1, 3)})

    agg = cvs.raster(xr_array, downsample_method='max', nan_value=9999)
    expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    assert np.allclose(agg.data, expected)
    assert agg.data.dtype.kind == 'i'
    assert np.allclose(agg.x.values, np.array([1/60., 1/20., 1/12.]))
    assert np.allclose(agg.y.values, np.array([1/60., 1/20., 1/12.]))


def test_raster_single_pixel_range_with_padding():
    """
    Ensure that canvas range covering a single pixel and small area
    beyond the defined data ranges is handled correctly.
    """

    # The .301 value ensures that one pixel covers the edge of the input extent
    cvs = ds.Canvas(plot_height=4, plot_width=6, x_range=(-0.5, 0.25), y_range=(-.5, 0.301))
    cvs2 = ds.Canvas(plot_height=4, plot_width=6, x_range=(-0.5, 0.25), y_range=(-.5, 0.3))
    array = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype='f')
    xr_array = xr.DataArray(array, dims=['y', 'x'],
                            coords={'x': np.linspace(0.125, .875, 4),
                                    'y': np.linspace(0.125, 0.625, 3)})
    agg = cvs.raster(xr_array, downsample_method='max', nan_value=np.nan)
    agg2 = cvs2.raster(xr_array, downsample_method='max', nan_value=np.nan)
    expected = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, 0, 0],
        [np.nan, np.nan, np.nan, np.nan, 0, 0]
    ])
    expected2 = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, 0, 0]
    ])

    assert np.allclose(agg.data, expected, equal_nan=True)
    assert np.allclose(agg2.data, expected2, equal_nan=True)
    assert agg.data.dtype.kind == 'f'
    assert np.allclose(agg.x.values, np.array([-0.4375, -0.3125, -0.1875, -0.0625, 0.0625, 0.1875]))
    assert np.allclose(agg.y.values, np.array([-0.399875, -0.199625,  0.000625,  0.200875]))


@dask_skip
@pytest.mark.parametrize(
    'in_size, out_size, agg',
    product(range(5, 8), range(2, 5),
            ['mean', 'min', 'max', 'first', 'last', 'var', 'std', 'mode']))
def test_raster_distributed_downsample(in_size, out_size, agg):
    """
    Ensure that distributed regrid is equivalent to regular regrid.
    """
    cvs = ds.Canvas(plot_height=out_size, plot_width=out_size)

    vs = np.linspace(-1, 1, in_size)
    xs, ys = np.meshgrid(vs, vs)
    arr = np.sin(xs*ys)

    darr = da.from_array(arr, (2, 2))
    coords = [('y', range(in_size)), ('x', range(in_size))]
    xr_darr = xr.DataArray(darr, coords=coords, name='z')
    xr_arr = xr.DataArray(arr, coords=coords, name='z')

    agg_arr = cvs.raster(xr_arr, agg=agg)
    agg_darr = cvs.raster(xr_darr, agg=agg)

    assert np.allclose(agg_arr.data, agg_darr.data.compute())
    assert np.allclose(agg_arr.x.values, agg_darr.x.values)
    assert np.allclose(agg_arr.y.values, agg_darr.y.values)


@dask_skip
@pytest.mark.parametrize('in_size, out_size', product(range(2, 5), range(7, 9)))
def test_raster_distributed_upsample(in_size, out_size):
    """
    Ensure that distributed regrid is equivalent to regular regrid.
    """
    cvs = ds.Canvas(plot_height=out_size, plot_width=out_size)

    vs = np.linspace(-1, 1, in_size)
    xs, ys = np.meshgrid(vs, vs)
    arr = np.sin(xs*ys)

    darr = da.from_array(arr, (2, 2))
    coords = [('y', range(in_size)), ('x', range(in_size))]
    xr_darr = xr.DataArray(darr, coords=coords, name='z')
    xr_arr = xr.DataArray(arr, coords=coords, name='z')

    agg_arr = cvs.raster(xr_arr, interpolate='nearest')
    agg_darr = cvs.raster(xr_darr, interpolate='nearest')

    assert np.allclose(agg_arr.data, agg_darr.data.compute())
    assert np.allclose(agg_arr.x.values, agg_darr.x.values)
    assert np.allclose(agg_arr.y.values, agg_darr.y.values)


@dask_skip
def test_raster_distributed_regrid_chunksize():
    """
    Ensure that distributed regrid respects explicit chunk size.
    """
    cvs = ds.Canvas(plot_height=2, plot_width=2)

    size = 4
    vs = np.linspace(-1, 1, size)
    xs, ys = np.meshgrid(vs, vs)
    arr = np.sin(xs*ys)

    darr = da.from_array(arr, (2, 2))
    xr_darr = xr.DataArray(darr, coords=[('y', range(size)), ('x', range(size))], name='z')

    agg_darr = cvs.raster(xr_darr, chunksize=(1, 1))

    assert agg_darr.data.chunksize == (1, 1)

@dask_skip
def test_resample_compute_chunksize():
    """
    Ensure chunksize computation is correct.
    """
    darr = da.from_array(np.zeros((100, 100)), (10, 10))

    mem_limited_chunksize = compute_chunksize(darr, 10, 10, max_mem=2000)
    assert mem_limited_chunksize == (2, 1)

    explicit_chunksize = compute_chunksize(darr, 10, 10, chunksize=(5, 4))
    assert explicit_chunksize == (5, 4)


@open_rasterio_available
def test_resample_methods(cvs):
    """Assert that an error is raised when incorrect upsample and/or downsample
    methods are provided to cvs.raster().
    """
    with open_rasterio(TEST_RASTER_PATH) as src:
        try:
            cvs.raster(src, upsample_method='santaclaus', downsample_method='toothfairy')
        except ValueError:
            pass
        else:
            assert False

        try:
            cvs.raster(src, upsample_method='honestlawyer')
        except ValueError:
            pass
        else:
            assert False

        try:
            cvs.raster(src, downsample_method='tenantfriendlylease')
        except ValueError:
            pass
        else:
            assert False


def test_raster_vs_points_coords():
    # Issue 1038.
    points = pd.DataFrame(data=dict(x=[2, 6, 8], y=[9, 7, 3]))
    raster = xr.DataArray(data=[[0.0, 1.0], [2.0, 3.0]], dims=("y", "x"),
                          coords=dict(x=[0, 9], y=[0, 11]))

    canvas = ds.Canvas(25, 15, x_range=(0, 10), y_range=(0, 5))
    agg_points = canvas.points(points, x="x", y="y")
    agg_raster = canvas.raster(raster)

    im_points = tf.shade(agg_points)
    im_raster = tf.shade(agg_raster)

    # Coordinates should be identical, not merely close.
    np.testing.assert_array_equal(im_points.coords["x"], im_raster.coords["x"])
    np.testing.assert_array_equal(im_points.coords["y"], im_raster.coords["y"])
