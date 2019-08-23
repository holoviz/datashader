import xarray as xa
import pytest

import datashader as ds
from datashader.spatial import viewshed

import numpy as np
import pandas as pd

width = 10
height = 5
x_range = (-20, 20)
y_range = (-20, 20)


df = pd.DataFrame({
   'x': [-10, -10, -4, -4, 1, 3, 7, 7, 7],
   'y': [-5, -10, -5, -5, 0, 5, 10, 10, 10]
})

cvs = ds.Canvas(plot_width=width,
                plot_height=height,
                x_range=x_range,
                y_range=y_range)

raster = cvs.points(df, x='x', y='y')


@pytest.mark.viewshed
def test_viewshed_invalid_x_view():
    x_view = 10
    y_view = 0
    with pytest.raises(Exception) as e_info:
        viewshed(raster=raster, x_view=x_view, y_view=y_view)
        assert e_info


@pytest.mark.viewshed
def test_viewshed_invalid_y_view():
    x_view = 0
    y_view = 5
    with pytest.raises(Exception) as e_info:
        viewshed(raster=raster, x_view=x_view, y_view=y_view)
        assert e_info


@pytest.mark.viewshed
def test_viewshed_invalid_x_range():
    x_view = 10
    y_view = 0
    with pytest.raises(Exception) as e_info:
        viewshed(raster=raster, x_view=x_view, y_view=y_view, x_range=(-1, 10))
        assert e_info


@pytest.mark.viewshed
def test_viewshed_invalid_y_range():
    x_view = 10
    y_view = 0
    with pytest.raises(Exception) as e_info:
        viewshed(raster=raster, x_view=x_view, y_view=y_view, y_range=(-1, 1))
        assert e_info


@pytest.mark.viewshed
def test_viewshed():
    x_view = 0
    y_view = 5
    v = viewshed(raster=raster, x_view=x_view, y_view=y_view)

    assert v.shape[0] == raster.values.shape[0]
    assert v.shape[1] == raster.values.shape[1]
    assert isinstance(v, xa.DataArray)
    assert isinstance(v.values, np.ndarray)
    assert type(v.values[0, 0]) == np.float64
