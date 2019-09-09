import pytest

from datashader.spatial import proximity
from datashader.spatial import great_circle_distance
import datashader as ds

import numpy as np
import pandas as pd
import xarray as xa

width = 10
height = 5


df = pd.DataFrame({
   'x': [-10, -10, -4, -4, 1, 3, 7, 7, 7],
   'y': [-5, -10, -5, -5, 0, 5, 10, 10, 10]
})

cvs = ds.Canvas(plot_width=width,
                plot_height=height,
                x_range=(-20, 20),
                y_range=(-20, 20))

raster = cvs.points(df, x='x', y='y')
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

    # number of non-zeros (target pixels) in original image
    # must be equal to the number of zeros (target pixels) in proximity matrix
    assert nonzeros_raster == zeros_default


@pytest.mark.proximity
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

    assert num_target == tv_zeros

@pytest.mark.proximity
def test_proximity_manhattan():

    # distance_metric SETTING
    dm_proximity = proximity(raster, distance_metric='MANHATTAN')

    # output must be an xarray DataArray
    assert isinstance(dm_proximity, xa.DataArray)
    assert isinstance(dm_proximity.values, np.ndarray)
    assert type(dm_proximity.values[0][0]) == np.float64
    assert dm_proximity.values.shape[0] == height
    assert dm_proximity.values.shape[1] == width

@pytest.mark.proximity
def test_proximity_distance_metric():

    # distance_metric SETTING
    dm_proximity = proximity(raster, distance_metric='GREAT_CIRCLE')

    # output must be an xarray DataArray
    assert isinstance(dm_proximity, xa.DataArray)
    assert isinstance(dm_proximity.values, np.ndarray)
    assert type(dm_proximity.values[0][0]) == np.float64
    assert dm_proximity.values.shape[0] == height
    assert dm_proximity.values.shape[1] == width


@pytest.mark.proximity
def test_greate_circle_invalid_x_coords():
    y1 = 0
    y2 = 0

    x1 = -181
    x2 = 0
    with pytest.raises(Exception) as e_info:
        great_circle_distance(x1, x2, y1, y2)
        assert e_info

    x1 = 181
    x2 = 0
    with pytest.raises(Exception) as e_info:
        great_circle_distance(x1, x2, y1, y2)
        assert e_info


@pytest.mark.proximity
def test_proximity_invalid_y_coords():

    x1 = 0
    x2 = 0

    y1 = -91
    y2 = 0
    with pytest.raises(Exception) as e_info:
        great_circle_distance(x1, x2, y1, y2)
        assert e_info

    y1 = 91
    y2 = 0
    with pytest.raises(Exception) as e_info:
        great_circle_distance(x1, x2, y1, y2)
        assert e_info
