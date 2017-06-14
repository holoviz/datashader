from os import path

import pytest
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
            agg = cvs.raster(src)
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
        assert agg.shape == (256, 512)
        assert agg is not None


def test_calc_res():
    """Assert that resolution is calculated correctly when using the xarray
    rasterio backend.
    """
    import rasterio
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
        xr_res = ds.utils.calc_res(src)
    with rasterio.open(TEST_RASTER_PATH) as src:
        rio_res = src.res
    assert np.allclose(xr_res, rio_res)


def test_calc_bbox():
    """Assert that bounding boxes are calculated correctly when using the xarray
    rasterio backend.
    """
    import rasterio
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
        xr_res = ds.utils.calc_res(src)
        xr_bounds = ds.utils.calc_bbox(src.x.values, src.y.values, xr_res)
    with rasterio.open(TEST_RASTER_PATH) as src:
        rio_bounds = src.bounds
    assert np.allclose(xr_bounds, rio_bounds)


def test_resample_methods():
    """Assert that an error is raised when incorrect upsample and/or downsample
    methods are provided to cvs.raster().
    """
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
        try:
            agg = cvs.raster(src, upsample_method='santaclaus', downsample_method='toothfairy')
        except ValueError:
            pass
        else:
            assert False

        try:
            agg = cvs.raster(src, upsample_method='honestlawyer')
        except ValueError:
            pass
        else:
            assert False

        try:
            agg = cvs.raster(src, downsample_method='tenantfriendlylease')
        except ValueError:
            pass
        else:
            assert False
