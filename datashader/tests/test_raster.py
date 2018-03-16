import pytest
rasterio = pytest.importorskip("rasterio")

from os import path

import datashader as ds
import xarray as xr
import numpy as np

BASE_PATH = path.split(__file__)[0]
DATA_PATH = path.abspath(path.join(BASE_PATH, 'data'))
TEST_RASTER_PATH = path.join(DATA_PATH, 'world.rgb.tif')

with xr.open_rasterio(TEST_RASTER_PATH) as src:
    res = ds.utils.calc_res(src)
    left, bottom, right, top = ds.utils.calc_bbox(src.x.values, src.y.values, res)
    cvs = ds.Canvas(plot_width=2,
                    plot_height=2,
                    x_range=(left, right),
                    y_range=(bottom, top))


def test_raster_aggregate_default():
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
        agg = cvs.raster(src)
        assert agg is not None


def test_raster_aggregate_nearest():
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
        agg = cvs.raster(src, upsample_method='nearest')
        assert agg is not None


@pytest.mark.skip('use_overviews opt no longer supported; may be re-implemented in the future')
def test_raster_aggregate_with_overviews():
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
        agg = cvs.raster(src, use_overviews=True)
        assert agg is not None


@pytest.mark.skip('use_overviews opt no longer supported; may be re-implemented in the future')
def test_raster_aggregate_without_overviews():
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
        agg = cvs.raster(src, use_overviews=False)
        assert agg is not None


def test_out_of_bounds_return_correct_size():
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
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


def test_partial_extent_returns_correct_size():
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
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


def test_partial_extent_with_layer_returns_correct_size():
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
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


def test_calc_res():
    """Assert that resolution is calculated correctly when using the xarray
    rasterio backend.
    """
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
        xr_res = ds.utils.calc_res(src)
    with rasterio.open(TEST_RASTER_PATH) as src:
        rio_res = src.res
    assert np.allclose(xr_res, rio_res)


def test_calc_bbox():
    """Assert that bounding boxes are calculated correctly when using the xarray
    rasterio backend.
    """
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
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
    array = np.array([[np.NaN, 1., 2., 3.], [4., np.NaN, 6., 7.], [8., 9., np.NaN, 11.]])
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
    xr_array = xr.DataArray(array, coords={'x': np.linspace(0, 1, 4), 'y': np.linspace(0, 1, 3)}, dims=['y', 'x'])

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
    xr_array = xr.DataArray(array, coords={'x': np.linspace(0, 1, 4), 'y': np.linspace(0, 1, 3)}, dims=['y', 'x'])

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

    cvs = ds.Canvas(plot_height=4, plot_width=4, x_range=(-0.5, 0.25), y_range=(-.5, 0.25))
    array = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype='f')
    xr_array = xr.DataArray(array, dims=['y', 'x'],
                            coords={'x': np.linspace(0, 1, 4),
                                    'y': np.linspace(0, 1, 3)})

    agg = cvs.raster(xr_array, downsample_method='max', nan_value=np.NaN)
    expected = np.array([[np.NaN, np.NaN, np.NaN, np.NaN], [np.NaN, 0, 0, 0],
                         [np.NaN, 0, 0, 0], [np.NaN, 0, 0, 0]])

    assert np.allclose(agg.data, expected, equal_nan=True)
    assert agg.data.dtype.kind == 'f'
    assert np.allclose(agg.x.values, np.array([-0.40625, -0.21875, -0.03125,  0.15625]))
    assert np.allclose(agg.y.values, np.array([-0.40625, -0.21875, -0.03125,  0.15625]))



def test_resample_methods():
    """Assert that an error is raised when incorrect upsample and/or downsample
    methods are provided to cvs.raster().
    """
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
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
