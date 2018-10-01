"""
This module contains geoscience-related transfer functions whose use is completely optional.

"""

from __future__ import division

from random import choice

import numpy as np
import datashader.transfer_functions as tf

from datashader.colors import rgb
from datashader.utils import ngjit
from xarray import DataArray

__all__ = ['mean', 'binary', 'slope', 'aspect', 'ndvi', 'hillshade', 'generate_terrain']


def hillshade(agg, azimuth, angle_altitude):
    """Illuminates 2D DataArray from specific azimuth and altitude.

    Parameters
    ----------
    agg : DataArray
    altitude : int, optional (default: 30)
        Altitude angle of the sun specified in degrees.
    azimuth : int, optional (default: 315)
        The angle between the north vector and the perpendicular projection
        of the light source down onto the horizon specified in degrees.
    cmap : list of colors or matplotlib.colors.Colormap, optional
        The colormap to use. Can be either a list of colors (in any of the
        formats described above), or a matplotlib colormap object.
        Default is `["lightgray", "black"]`
    how : str or callable, optional
        The hillshade method to use. Valid strings are 'mdow' [default],
        'simple'.
    alpha : int, optional
        Value between 0 - 255 representing the alpha value of pixels which contain
        data (i.e. non-nan values). Regardless of this value, `NaN` values are
        set to fully transparent.

    Returns
    -------
    Datashader Image

    """
    azimuth = 360.0 - azimuth
    x, y = np.gradient(agg.data)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.
    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)
    data = (shaded + 1) / 2
    return DataArray(data, attrs={'res':1})


@ngjit
def _horn_slope(data, cellsize, use_percent=True):
    out = np.zeros_like(data)
    rows, cols = data.shape
    if use_percent:
        for y in range(1, rows-1):
            for x in range(1, cols-1):
                a,b,c,d,f,g,h,i = [data[y-1, x-1], data[y, x-1], data[y+1, x-1],
                                     data[y-1, x], data[y+1, x],
                                     data[y-1, x+1], data[y, x+1], data[y+1, x+1]]
                dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize)
                dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize)
                out[y, x] = (dz_dx ** 2 + dz_dy ** 2) ** .5
    else:
        for y in range(1, rows-1):
            for x in range(1, cols-1):
                a,b,c,d,f,g,h,i = [data[y-1, x-1], data[y, x-1], data[y+1, x-1],
                                     data[y-1, x], data[y+1, x],
                                     data[y-1, x+1], data[y, x+1], data[y+1, x+1]]  #NOQA
                dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize)
                dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize)
                p = (dz_dx ** 2 + dz_dy ** 2) ** .5
                out[y, x] = np.arctan(p) * 180 / np.pi

    return out


def slope(agg, units='percent'):
    """Returns slope of input aggregate in percent.

    Parameters
    ----------
    agg : DataArray
    units : str, optional (default: percent)
        The units of the return values. options `percent`, or `degrees`.

    Returns
    -------
    data: DataArray
    """

    if not isinstance(agg, DataArray):
        raise TypeError("agg must be instance of DataArray")

    if units not in ('percent', 'degree'):
        raise ValueError('Invalid slope units: options (percent, degree)')

    if not agg.attrs.get('res'):
        #TODO: maybe monkey-patch a "res" attribute valueing unity is reasonable
        raise ValueError('input xarray must have numeric `res` attr.')

    use_percent = units == 'percent'
    slope_agg = _horn_slope(agg.data,
                            agg.attrs['res'],
                            use_percent=use_percent)

    return DataArray(slope_agg,
                     name='slope',
                     coords=agg.coords,
                     dims=agg.dims,
                     attrs=agg.attrs)

@ngjit
def _ndvi(nir_data, red_data):
    out = np.zeros_like(nir_data)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            red = red_data[y, x]
            soma = nir + red
            if soma != 0:
                out[y, x] = (nir - red) / soma
    return out

def ndvi(nir_agg, red_agg, units='percent'):
    """Returns Normalized Difference Vegetation Index (NDVI).

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band data
    red_agg : DataArray
        red band data

    Returns
    -------
    data: DataArray
    """

    if not isinstance(nir_agg, DataArray):
        raise TypeError("nir_agg must be instance of DataArray")

    if not isinstance(red_agg, DataArray):
        raise TypeError("red_agg must be instance of DataArray")

    if not red_agg.shape == nir_agg.shape:
        raise ValueError("red_agg and nir_agg expected to have equal shapes")

    return DataArray(_ndvi(nir_agg.data, red_agg.data),
                     attrs=nir_agg.attrs)

@ngjit
def _horn_aspect(data):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            a,b,c,d,f,g,h,i = [data[y-1, x-1], data[y, x-1], data[y+1, x-1],
                                 data[y-1, x], data[y+1, x],
                                 data[y-1, x+1], data[y, x+1], data[y+1, x+1]]
            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g))
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c))
            aspect = np.arctan2(dz_dy, -dz_dx) * 180 / np.pi
            out[y, x] = aspect + 180
    return out


def aspect(agg):
    """Returns downward slope direction in compass degrees (0 - 360) with 0 at 12 o'clock.

    Parameters
    ----------
    agg : DataArray

    Returns
    -------
    data: DataArray
    """

    if not isinstance(agg, DataArray):
        raise TypeError("agg must be instance of DataArray")

    return DataArray(_horn_aspect(agg.data),
                     dims=['y', 'x'],
                     attrs=agg.attrs)


def color_values(agg, color_key, alpha=255):

    def _convert_color(c):
        r, g, b = rgb(c)
        return np.array([r, g, b, alpha]).astype(np.uint8).view(np.uint32)[0]

    _converted_colors = {k: _convert_color(v) for k, v in color_key.items()}
    f = np.vectorize(lambda v: _converted_colors.get(v, 0))
    return tf.Image(f(agg.data))


@ngjit
def _binary(data, values):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for x in range(0, rows):
        for y in range(0, cols):
            if data[y, x] in values:
                out[y, x] = True
            else:
                out[y, x] = False
    return out

def binary(agg, values):
    return DataArray(_binary(agg.data, values),
                     dims=['y', 'x'],
                     attrs=agg.attrs)


@ngjit
def _mean(data):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            a,b,c,d,e,f,g,h,i = [data[y-1, x-1], data[y, x-1], data[y+1, x-1],
                                 data[y-1, x],   data[y, x],   data[y+1, x],
                                 data[y-1, x+1], data[y, x+1], data[y+1, x+1]]

            out[y, x] = (a+b+c+d+e+f+g+h+i) / 9
    return out

def mean(agg):
    """
    Returns Mean filtered array using a 3x3 window

    Parameters
    ----------
    agg : DataArray

    Returns
    -------
    data: DataArray
    """
    return DataArray(_mean(agg.data), dims=['y', 'x'], attrs=agg.attrs)


def generate_terrain(width, height, seed=10, iterations=10, extrusion_factor=200):
    """
    Generates a pseudo-random terrain which can be helpful for testing raster functions

    This was heavily inspired by Michael McHugh's 2016 PyCon Canada talk:
    https://www.youtube.com/watch?v=O33YV4ooHSo

    Perlin noise is used to seed to terrain taken from here, but scaled from 0 - 1
    and was written by Paul Panzer and is available here:
    https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy

    Parameters
    ----------
    width : int
    height : int
    seed : seed for random number generator
    iterations : number of noise iterations

    Returns
    -------
    terrain: DataArray
    """
    data = _gen_terrain(width, height, seed, iterations)
    data = _add_texture(data, width, height, count=int(1e6), size=.05, mind=0.33, maxd=3)
    data[data < .3] = 0  # create water
    return DataArray(data * extrusion_factor, attrs={'res':1})

def _gen_terrain(width, height, seed, iterations):
    amp_factors = list(range(10))
    height_map = None
    linx = np.linspace(0, 1, width, endpoint=False)
    liny = np.linspace(0, 1, height, endpoint=False)
    x, y = np.meshgrid(linx, liny)
    for i in reversed(range(iterations)):
        noise = _perlin(x * choice(amp_factors), y * choice(amp_factors), seed=seed + (i % 500))
        noise += _perlin(x, y, seed=seed + (i % 400)) ** 2

        if height_map is None:
            height_map = noise
        else:
            height_map += noise

    return height_map

def _add_texture(data, width, height, count=100,
                 seed=5, size=.2, mind=.33, maxd=.87):
    linx = range(width)
    liny = range(height)
    tree_xs = np.random.choice(linx, count).tolist()
    tree_ys = np.random.choice(liny, count).tolist()
    trees = _finish_trees(data, list(zip(tree_xs, tree_ys)), size, mind, maxd)
    return data + _mean(_mean(_mean(trees)))

@ngjit
def _finish_trees(data, locs, size, mind, maxd):
    out = np.zeros_like(data)
    for x, y in locs:
        if data[y, x] >= mind and data[y, x] < maxd:
            out[y, x] = size
    return out


def _perlin(x, y, seed=0):

    @ngjit
    def lerp(a, b, x):
        return a + x * (b-a)

    @ngjit
    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def gradient(h,x,y):
        vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
        g = vectors[h%4]
        return g[:,:,0] * x + g[:,:,1] * y

    # permutation table
    np.random.seed(seed)
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()

    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)

    # internal coordinates
    xf = x - xi
    yf = y - yi

    # fade factors
    u = fade(xf)
    v = fade(yf)

    # noise components
    n00 = gradient(p[p[xi]+yi], xf, yf)
    n01 = gradient(p[p[xi]+yi+1], xf, yf-1)
    n11 = gradient(p[p[xi+1]+yi+1], xf-1, yf-1)
    n10 = gradient(p[p[xi+1]+yi], xf-1, yf)

    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    a = lerp(x1, x2, v)
    return a

