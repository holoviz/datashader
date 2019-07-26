import datashader as ds
import xarray as xr
import numpy as np
import pytest

from datashader import geo

W = 25
H = 30

X_RANGE = (0, 500)
Y_RANGE = (0, 500)

csv = ds.Canvas(x_range=X_RANGE, y_range=Y_RANGE,
                plot_width=W, plot_height=H)
terrain = geo.generate_terrain(csv)

def _raster_aggregate_default():
    return terrain
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
    assert pytest.approx(da_aspect.data.max(), .1) == 360.
    assert pytest.approx(da_aspect.data.min(), .1) == 0.

def test_hillshade_simple_transfer_function():
    """
    Assert Simple Hillshade transfer function
    """
    da_gaussian = xr.DataArray(data_gaussian)
    da_gaussian_shade = geo.hillshade(da_gaussian)

    assert da_gaussian_shade.mean() > 0
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


def test_generate_terrain():
    csv = ds.Canvas(x_range=X_RANGE, y_range=Y_RANGE,
                    plot_width=W, plot_height=H)
    terrain = geo.generate_terrain(csv)
    assert terrain is not None


def test_bump():
    bumps = geo.bump(20, 20)
    assert bumps is not None


def test_proximity():

    x_range = (-50, 50)
    y_range = (-50, 50)
    width = 100
    height = 50
    dtype = np.uint8
    result = rasterize(fc, height=height, width=width, x_range=x_range, y_range=y_range, dtype=dtype)

    raster_image = result.values
    nonzeros_raster = np.count_nonzero(raster_image)
    zeros_raster = width*height - nonzeros_raster

    # DEFAULT SETTINGS
    # proximity(img, max_distance=None, target_values=[], dist_units=PIXEL, no_data=np.nan):
    default_proximity = proximity(raster_image)
    zeros_default = (default_proximity == 0).sum()

    assert isinstance(default_proximity, np.ndarray)
    assert type(default_proximity[0][0]) == np.float64
    assert default_proximity.shape[0] == 50
    assert default_proximity.shape[1] == 100
    # number of non-zeros (target pixels) in original image
    # must be equal to the number of zeros (target pixels) in proximity matrix
    assert nonzeros_raster == zeros_default

    # MAX_DISTANCE SETTING
    # pixels that beyond max_distance will be set to np.nan
    md_proximity = proximity(raster_image, max_distance=0)
    md_nan = np.isnan(md_proximity).sum()
    md_zeros = (md_proximity == 0).sum()

    # number of non-zeros (target pixels) in original image
    # must be equal to the number of zeros (target pixels) in proximity matrix
    assert nonzeros_raster == md_zeros
    # set max_distance to 0, all distance will be np.nan
    # number of zeros (non target pixels) in original image
    # must be equal to the number of np.nan (non target pixels) in proximity matrix
    assert zeros_raster == md_nan

    max_distance = 3
    # pixels that beyond max_distance will be set to np.nan
    md_proximity = proximity(raster_image, max_distance=max_distance)
    max_dist = np.nanmax(md_proximity)
    md_zeros = (md_proximity == 0).sum()

    assert nonzeros_raster == md_zeros
    assert max_dist <= max_distance

    # MIN_DISTANCE SETTING
    # pixels that less than min_distance will be set to np.nan
    min_distance = 2
    min_d_proximity = proximity(raster_image, min_distance=min_distance)
    min_dist = np.nanmin(min_d_proximity[np.nonzero(min_d_proximity)])
    min_d_zeros = (min_d_proximity == 0).sum()

    assert nonzeros_raster == min_d_zeros
    # min_distance must not exceed the min value of the proximity matrix
    assert min_dist >= min_distance

    max_distance = 3
    # pixels that beyond max_distance will be set to np.nan
    md_proximity = proximity(raster_image, max_distance=max_distance)
    max_dist = np.nanmax(md_proximity)
    md_zeros = (md_proximity == 0).sum()

    assert nonzeros_raster == md_zeros
    # the max value of the proximity matrix must not exceed max_distance
    assert max_dist <= max_distance

    # TARGET VALUES SETTING
    target_values = [2, 3]
    num_target = (raster_image == 2).sum() + (raster_image == 3).sum()
    tv_proximity = proximity(raster_image, target_values=target_values)
    tv_zeros = (tv_proximity == 0).sum()

    assert num_target == tv_zeros

    # NODATA SETTING
    max_distance = 3
    no_data = -1
    nd_proximity = proximity(raster_image, max_distance=max_distance, no_data=no_data)
    max_dist = np.nanmax(md_proximity)
    nd_zeros = (nd_proximity == 0).sum()

    # pixels that beyond max_distance will be set to no_data
    num_nodata = (nd_proximity == no_data).sum()
    # number of nan if no_data is not set
    md_nan = np.isnan(md_proximity).sum()

    assert max_dist <= max_distance
    assert nonzeros_raster == nd_zeros
    assert num_nodata == md_nan
