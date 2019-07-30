import pytest

from datashader.spatial import proximity

import numpy as np
import xarray as xa

width = 10
height = 5

agg_values = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                       [0, 0, 0, 0, 3, 5, 4, 0, 0, 3],
                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 2]],
                       dtype=np.uint8)

raster = xa.DataArray(agg_values)
raster_image = raster.values
nonzeros_raster = np.count_nonzero(raster_image)
zeros_raster = width * height - nonzeros_raster


@pytest.mark.proximity
def test_proximity_default():

    # DEFAULT SETTINGS
    # proximity(img, max_distance=None, target_values=[], dist_units=PIXEL,
    #           nodata=np.nan)
    default_proximity = proximity(raster)
    default_proximity_img = default_proximity.values
    zeros_default = (default_proximity_img == 0).sum()

    # output must be an xarray DataArray
    assert isinstance(default_proximity, xa.DataArray)
    assert isinstance(default_proximity.values, np.ndarray)
    assert type(default_proximity.values[0][0]) == np.float64
    assert default_proximity.values.shape[0] == height
    assert default_proximity.values.shape[1] == width

    # attributes of the xarray DataArray
    assert default_proximity.attrs['min_distance'] == 0.0
    assert default_proximity.attrs['max_distance'] == width + height
    assert len(default_proximity.attrs['target_values']) == 0
    assert np.isnan(default_proximity.attrs['nodata_value'])
    assert default_proximity.attrs['distance_metric'] == 'square distance'

    # number of non-zeros (target pixels) in original image
    # must be equal to the number of zeros (target pixels) in proximity matrix
    assert nonzeros_raster == zeros_default


def test_proximity_max_distance():

    # MAX_DISTANCE SETTING
    # pixels that beyond max_distance will be set to np.nan
    max_distance = 0
    md_proximity = proximity(raster, max_distance=max_distance)
    md_proximity_img = md_proximity.values
    md_nan = np.isnan(md_proximity_img).sum()
    md_zeros = (md_proximity_img == 0).sum()

    # output must be an xarray DataArray
    assert isinstance(md_proximity, xa.DataArray)
    assert isinstance(md_proximity.values, np.ndarray)
    assert type(md_proximity.values[0][0]) == np.float64
    assert md_proximity.values.shape[0] == height
    assert md_proximity.values.shape[1] == width

    # attributes of the xarray DataArray
    assert md_proximity.attrs['min_distance'] == 0.0
    assert md_proximity.attrs['max_distance'] == max_distance
    assert len(md_proximity.attrs['target_values']) == 0
    assert np.isnan(md_proximity.attrs['nodata_value'])
    assert md_proximity.attrs['distance_metric'] == 'square distance'

    # number of non-zeros (target pixels) in original image
    # must be equal to the number of zeros (target pixels) in proximity matrix
    assert nonzeros_raster == md_zeros
    # set max_distance to 0, all distance will be np.nan
    # number of zeros (non target pixels) in original image
    # must be equal to the number of np.nan (non target pixels)
    # in proximity matrix
    assert zeros_raster == md_nan

    max_distance = 3
    # pixels that beyond max_distance will be set to np.nan
    md_proximity = proximity(raster, max_distance=max_distance)
    md_proximity_img = md_proximity.values
    max_dist = np.nanmax(md_proximity_img)
    md_zeros = (md_proximity_img == 0).sum()

    # attributes of the xarray DataArray
    assert md_proximity.attrs['max_distance'] == max_distance

    assert nonzeros_raster == md_zeros
    assert max_dist <= max_distance


def test_proximity_min_distance():

    # MIN_DISTANCE SETTING
    # pixels that less than min_distance will be set to np.nan
    min_distance = 2
    min_d_proximity = proximity(raster, min_distance=min_distance)
    min_d_proximity_img = min_d_proximity.values

    min_dist = np.nanmin(min_d_proximity_img[np.nonzero(min_d_proximity_img)])
    min_d_zeros = (min_d_proximity_img == 0).sum()

    # output must be an xarray DataArray
    assert isinstance(min_d_proximity, xa.DataArray)
    assert isinstance(min_d_proximity.values, np.ndarray)
    assert type(min_d_proximity.values[0][0]) == np.float64
    assert min_d_proximity.values.shape[0] == height
    assert min_d_proximity.values.shape[1] == width

    # attributes of the xarray DataArray
    assert min_d_proximity.attrs['min_distance'] == min_distance
    assert min_d_proximity.attrs['max_distance'] == height + width
    assert len(min_d_proximity.attrs['target_values']) == 0
    assert np.isnan(min_d_proximity.attrs['nodata_value'])
    assert min_d_proximity.attrs['distance_metric'] == 'square distance'

    assert nonzeros_raster == min_d_zeros
    # min_distance must not exceed the min value of the proximity matrix
    assert min_dist >= min_distance

    max_distance = 3
    # pixels that beyond max_distance will be set to np.nan
    md_proximity = proximity(raster, max_distance=max_distance)
    md_proximity_img = md_proximity.values

    max_dist = np.nanmax(md_proximity_img)
    md_zeros = (md_proximity_img == 0).sum()

    assert nonzeros_raster == md_zeros
    # the max value of the proximity matrix must not exceed max_distance
    assert max_dist <= max_distance


def test_proximity_target_value():

    # TARGET VALUES SETTING
    target_values = [2, 3]
    num_target = (raster == 2).sum() + (raster == 3).sum()
    tv_proximity = proximity(raster, target_values=target_values)
    tv_proximity_img = tv_proximity.values
    tv_zeros = (tv_proximity_img == 0).sum()

    # output must be an xarray DataArray
    assert isinstance(tv_proximity, xa.DataArray)
    assert isinstance(tv_proximity.values, np.ndarray)
    assert type(tv_proximity.values[0][0]) == np.float64
    assert tv_proximity.values.shape[0] == height
    assert tv_proximity.values.shape[1] == width

    # attributes of the xarray DataArray
    assert tv_proximity.attrs['min_distance'] == 0
    assert tv_proximity.attrs['max_distance'] == height + width
    assert len(tv_proximity.attrs['target_values']) == 2
    assert tv_proximity.attrs['target_values'][0] == 2
    assert tv_proximity.attrs['target_values'][1] == 3
    assert np.isnan(tv_proximity.attrs['nodata_value'])
    assert tv_proximity.attrs['distance_metric'] == 'square distance'

    assert num_target == tv_zeros


def test_proximity_nodata():

    # NODATA SETTING
    max_distance = 3
    # pixels that beyond max_distance will be set to np.nan
    md_proximity = proximity(raster, max_distance=max_distance)

    nodata = -1
    nd_proximity = proximity(raster, max_distance=max_distance,
                             nodata=nodata)
    nd_proximity_img = nd_proximity.values
    max_dist = np.nanmax(md_proximity.values)
    nd_zeros = (nd_proximity_img == 0).sum()

    # output must be an xarray DataArray
    assert isinstance(nd_proximity, xa.DataArray)
    assert isinstance(nd_proximity.values, np.ndarray)
    assert type(nd_proximity.values[0][0]) == np.float64
    assert nd_proximity.values.shape[0] == height
    assert nd_proximity.values.shape[1] == width

    # attributes of the xarray DataArray
    assert nd_proximity.attrs['min_distance'] == 0
    assert nd_proximity.attrs['max_distance'] == max_distance
    assert len(nd_proximity.attrs['target_values']) == 0
    assert nd_proximity.attrs['nodata_value'] == nodata
    assert nd_proximity.attrs['distance_metric'] == 'square distance'

    # pixels that beyond max_distance will be set to nodata
    num_nodata = (nd_proximity_img == nodata).sum()
    # number of nan if nodata is not set
    md_nan = np.isnan(md_proximity.values).sum()

    assert max_dist <= max_distance
    assert nonzeros_raster == nd_zeros
    assert num_nodata == md_nan
