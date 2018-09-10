# import pytest

from os import path

import datashader as ds
import xarray as xr
import numpy as np

from datashader import geo #mean, binary, slope, aspect, ndvi, hillshade

# use landsat data from sample data
BASE_PATH = path.split(__file__)[0]
DATA_PATH = path.abspath(path.join(BASE_PATH, 'data'))
# TEST_RASTER_PATH = path.join(DATA_PATH, 'landsat.tif???') # TODO: update me!
TEST_RASTER_PATH = path.join(DATA_PATH, 'world.rgb.tif')


# -----
# Utils
#
with xr.open_rasterio(TEST_RASTER_PATH) as src:
    res = ds.utils.calc_res(src)
    left, bottom, right, top = ds.utils.calc_bbox(src.x.values, src.y.values, res)
    cvs = ds.Canvas(plot_width=2,
                    plot_height=2,
                    x_range=(left, right),
                    y_range=(bottom, top))

def _raster_aggregate_default():
    with xr.open_rasterio(TEST_RASTER_PATH) as src:
        agg = cvs.raster(src)
        assert agg is not None
    return agg

# Take a numpy array and make it randomly sparse (50% -> zeros)
#
def _do_sparse_array(data_array):
    import random
    indx = list(zip(*np.where(data_array)))
    pos = random.sample(range(data_array.size), data_array.size//2)
    indx = np.asarray(indx)[pos]
    r = indx[:,0]
    c = indx[:,1]
    data_half = data_array.copy()
    data_half[r,c] = 0
    return data_half

def _do_gaussian_array():
    _x = np.linspace(0, 50, 101)
    _y = _x.copy()
    _mean = 25
    _sdev = 5
    X,Y = np.meshgrid(_x, _y, sparse=True)
    x_fac = -np.power(X-_mean, 2)
    y_fac = -np.power(Y-_mean, 2)
    gaussian = np.exp((x_fac+y_fac)/(2*_sdev**2)) / (2.5*_sdev)
    return gaussian
#
# -----

data_random = np.random.random_sample((100,100))
data_random_sparse = _do_sparse_array(data_random)
data_gaussian = _do_gaussian_array()


def test_mean_transfer_function():
    da = xr.DataArray(data_random)
    da_mean = geo.mean(da)
    assert da.shape == da_mean.shape

    # Overall mean value should be the same as the original array.
    # Considering the default behaviour to 'mean' is to pad the borders
    # with zeros, the mean value of the filtered array will be slightly
    # smaller (considering 'data_random' is positive).
    assert da_mean.mean() <= data_random.mean()

    # And if we pad the borders with the original values, we should have a
    # 'mean' filtered array with _mean_ value very similar to the original one.
    da_mean[0,:] = data_random[0,:]
    da_mean[-1,:]= data_random[-1,:]
    da_mean[:,0] = data_random[:,0]
    da_mean[:,-1]= data_random[:,-1]
    assert abs(da_mean.mean() - data_random.mean()) < 10**-3

def test_slope_transfer_function():
    """
    Assert slope transfer function
    """
    da = xr.DataArray(data_gaussian, attrs={'res':1})
    da_slope = geo.slope(da)
    assert da.shape == da_slope.shape

    assert da_slope.sum() > 0

    # In the middle of the array, there is the maximum of the gaussian;
    # And there the slope must be zero.
    _imax = np.where(da == da.max())
    assert da_slope[_imax] == 0

def test_aspect_transfer_function():
    """
    Assert aspect transfer function
    """
    da = xr.DataArray(data_gaussian, attrs={'res':1})
    da_aspect = geo.aspect(da)
    assert da.shape == da_aspect.shape

    # Running clockwise, from [0:360] degrees, with origin the the vertical axis
    y_mid = data_gaussian.shape[0]//2
    x_mid = data_gaussian.shape[1]//2

    print(da_aspect)
    # middle-top
    assert da_aspect[1,x_mid] == 0 or da_aspect[1,x_mid] == 360
    # right-middle
    assert da_aspect[y_mid,-2] == 90
    # middle-bottom
    assert da_aspect[-2,x_mid] == 180
    # left-middle
    assert da_aspect[y_mid,1] == 270
    # top-right
    assert da_aspect[1,-2] == 45
    # bottom-right
    assert da_aspect[-2,-2] == 135
    # bottom-left
    assert da_aspect[-2,1] == 225
    # top-left
    assert da_aspect[1,1] == 315

def test_hillshade_simple_transfer_function():
    """
    Assert Simple Hillshade transfer function
    """
    da_gaussian = xr.DataArray(data_gaussian)
    da_gaussian_shade = geo.hillshade(da_gaussian, how='simple', out_type='data')

    assert da_gaussian_shade.mean() > 0
    assert da_gaussian_shade[1,1] == 0
    assert da_gaussian_shade[60,60] > 0

def test_ndvi_transfer_function():
    """
    Assert aspect transfer function
    """
    _x = np.mgrid[1:0:21j]
    a,b = np.meshgrid(_x,_x)
    red = a*b
    nir = (a*b)[::-1,::-1]

    da_red = xr.DataArray(red)
    da_nir = xr.DataArray(nir)
    da_ndvi = geo.ndvi(da_nir, da_red)

    assert da_ndvi[0,0] == -1
    assert da_ndvi[-1,-1] == 1
    assert da_ndvi[5,10] == da_ndvi[10,5] == -0.5
    assert da_ndvi[15,10] == da_ndvi[10,15] == 0.5
