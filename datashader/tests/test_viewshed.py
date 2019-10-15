import xarray as xa
import pytest

import datashader as ds
from datashader.spatial import viewshed

import numpy as np
import pandas as pd

# create an empty image of size 5*5
H = 5
W = 5

canvas = ds.Canvas(plot_width=W, plot_height=H,
                   x_range=(-20, 20), y_range=(-20, 20))

empty_df = pd.DataFrame({
   'x': np.random.normal(.5, 1, 0),
   'y': np.random.normal(.5, 1, 0)
})
empty_agg = canvas.points(empty_df, 'x', 'y')

# coordinates
xs = empty_agg.coords['x'].values
ys = empty_agg.coords['x'].values

# define some values for observer's elevation to test
OBS_ELEVS = [-1, 0, 1]

TERRAIN_ELEV_AT_VP = [-1, 0, 1]

@pytest.mark.viewshed
def test_viewshed_invalid_x_view():
    OBSERVER_X = xs[0] - 1
    OBSERVER_Y = 0
    with pytest.raises(Exception) as e_info: # NOQA
        viewshed(raster=empty_agg, x=OBSERVER_X, y=OBSERVER_Y,
                 observer_elev=10)


@pytest.mark.viewshed
def test_viewshed_invalid_y_view():
    OBSERVER_X = 0
    OBSERVER_Y = ys[-1] + 1
    with pytest.raises(Exception) as e_info: # NOQA
        viewshed(raster=empty_agg, x=OBSERVER_X, y=OBSERVER_Y,
                 observer_elev=10)


@pytest.mark.viewshed
def test_viewshed_output_properties():
    for obs_elev in OBS_ELEVS:
        OBSERVER_X = xs[0]
        OBSERVER_Y = ys[0]
        v = viewshed(raster=empty_agg, x=OBSERVER_X, y=OBSERVER_Y,
                     observer_elev=obs_elev)

        assert v.shape[0] == empty_agg.shape[0]
        assert v.shape[1] == empty_agg.shape[1]
        assert isinstance(v, xa.DataArray)
        assert isinstance(v.values, np.ndarray)
        assert type(v.values[0, 0]) == np.float64


@pytest.mark.viewshed
def test_viewshed():

    # check if a matrix is symmetric
    def check_symmetric(matrix, rtol=1e-05, atol=1e-08):
        return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

    def get_matrices(y, x, height, width):
        # indexing 0 1 ... height-1 and 0 1 ... width-1
        height = height - 1
        width = width - 1

        # find first matrix's diagonal
        tmp = min(y, x)
        f_top_y, f_left_x = y - tmp, x - tmp

        tmp = min(height - y, width - x)
        f_bottom_y, f_right_x = y + tmp, x + tmp

        # find second matrix's antidiagonal
        tmp = min(y, width - x)
        s_top_y, s_right_x = y - tmp, x + tmp

        tmp = min(height - y, x)
        s_bottom_y, s_left_x = y + tmp, x - tmp

        return (f_top_y, f_left_x, f_bottom_y + 1, f_right_x + 1), \
               (s_top_y, s_left_x, s_bottom_y + 1, s_right_x + 1)

    # test on 3 scenarios:
    #   empty image.
    #   image with all 0s, except 1 cell with a negative value.
    #   image with all 0s, except 1 cell with a positive value.

    # for each scenario:
    #   if not empty image,
    #      observer is located at the same position as the non zero value.
    #   observer elevation can be: negative, zero, or positive.

    # assertion:
    #   angle at viewpoint is always 180.
    #   when the observer is above the terrain, all cells are visible.
    #   the symmetric property of observer's visibility.

    for obs_elev in OBS_ELEVS:
        for elev_at_vp in TERRAIN_ELEV_AT_VP:
            for col_id, x in enumerate(xs):
                for row_id, y in enumerate(ys):
                    empty_agg.values[row_id, col_id] = elev_at_vp
                    v = viewshed(raster=empty_agg, x=x, y=y,
                                 observer_elev=obs_elev)
                    # angle at viewpoint is always 180
                    assert v[row_id, col_id] == 180

                    if obs_elev + elev_at_vp >= 0 and \
                            obs_elev >= abs(elev_at_vp):
                        # all cells are visible
                        assert (v.values > -1).all()

                    b1, b2 = get_matrices(row_id, col_id, H, W)
                    m1 = v.values[b1[0]:b1[2], b1[1]:b1[3]]
                    m2 = v.values[b2[0]:b2[2], b2[1]:b2[3]]

                    assert check_symmetric(m1)
                    assert check_symmetric(m2[::-1])

                    # empty image for next uses
                    empty_agg.values[row_id, col_id] = 0
