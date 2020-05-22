"""
This module contains geoscience-related transfer functions whose use is completely optional.

"""

from __future__ import division, absolute_import

import warnings

import pandas as pd
import numpy as np
import datashader.transfer_functions as tf

from datashader import Canvas
from datashader.colors import rgb
from datashader.utils import (
    VisibleDeprecationWarning, ngjit, lnglat_to_meters   # noqa (API import)
)
from xarray import DataArray

__all__ = ['mean', 'binary', 'slope', 'aspect', 'ndvi', 'hillshade', 'generate_terrain',
           'lnglat_to_meters']

warnings.warn(
    "The datashader.geo module is deprecated as of version 0.11.0. "
    "Its contents have migrated to the xarray_spatial library, "
    "github.com/makepath/xarray-spatial.",
    VisibleDeprecationWarning
)


# TODO: add optional name parameter `name='hillshade'`
def hillshade(agg, azimuth=225, angle_altitude=25):
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

    Notes:
    ------
    Algorithm References:
     - http://geoexamples.blogspot.com/2014/03/shaded-relief-images-using-gdal-python.html
    """
    azimuth = 360.0 - azimuth
    x, y = np.gradient(agg.data)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.
    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)
    data = (shaded + 1) / 2
    return DataArray(data, name='hillshade', dims=agg.dims, coords=agg.coords, attrs=agg.attrs)


@ngjit
def _horn_slope(data, cellsize):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            a = data[y+1, x-1]
            b = data[y+1, x]
            c = data[y+1, x+1]
            d = data[y, x-1]
            f = data[y, x+1]
            g = data[y-1, x-1]
            h = data[y-1, x]
            i = data[y-1, x+1]
            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize)
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize)
            p = (dz_dx * dz_dx + dz_dy * dz_dy) ** .5
            out[y, x] = np.arctan(p) * 57.29578
    return out


# TODO: add optional name parameter `name='slope'`
def slope(agg):
    """Returns slope of input aggregate in degrees.

    Parameters
    ----------
    agg : DataArray

    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
     - http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm
     - Burrough, P. A., and McDonell, R. A., 1998. Principles of Geographical Information Systems (Oxford University Press, New York), pp 406
    """

    if not isinstance(agg, DataArray):
        raise TypeError("agg must be instance of DataArray")

    if not agg.attrs.get('res'):
        #TODO: maybe monkey-patch a "res" attribute valueing unity is reasonable
        raise ValueError('input xarray must have numeric `res` attr.')

    slope_agg = _horn_slope(agg.data, agg.attrs['res'])

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

            if nir == red:  # cover zero divison case
                continue

            soma = nir + red
            out[y, x] = (nir - red) / soma
    return out

# TODO: add optional name parameter `name='ndvi'`
def ndvi(nir_agg, red_agg):
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

    Notes:
    ------
    Algorithm References:
     - http://ceholden.github.io/open-geo-tutorial/python/chapter_2_indices.html
    """

    if not isinstance(nir_agg, DataArray):
        raise TypeError("nir_agg must be instance of DataArray")

    if not isinstance(red_agg, DataArray):
        raise TypeError("red_agg must be instance of DataArray")

    if not red_agg.shape == nir_agg.shape:
        raise ValueError("red_agg and nir_agg expected to have equal shapes")

    return DataArray(_ndvi(nir_agg.data, red_agg.data),
                     name='ndvi',
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)

@ngjit
def _horn_aspect(data):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):

            a = data[y+1, x-1]
            b = data[y+1, x]
            c = data[y+1, x+1]
            d = data[y, x-1]
            f = data[y, x+1]
            g = data[y-1, x-1]
            h = data[y-1, x]
            i = data[y-1, x+1]

            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / 8
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / 8

            aspect = np.arctan2(dz_dy, -dz_dx) * 57.29578  # (180 / pi)

            if aspect < 0:
                out[y, x] = 90.0 - aspect
            elif aspect > 90.0:
                out[y, x] = 360.0 - aspect + 90.0
            else:
                out[y, x] = 90.0 - aspect

    return out


# TODO: add optional name parameter `name='aspect'`
def aspect(agg):
    """Returns downward slope direction in compass degrees (0 - 360) with 0 at 12 o'clock.

    Parameters
    ----------
    agg : DataArray

    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
     - http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm#ESRI_SECTION1_4198691F8852475A9F4BC71246579FAA
     - Burrough, P. A., and McDonell, R. A., 1998. Principles of Geographical Information Systems (Oxford University Press, New York), pp 406
    """

    if not isinstance(agg, DataArray):
        raise TypeError("agg must be instance of DataArray")

    return DataArray(_horn_aspect(agg.data),
                     name='aspect',
                     dims=agg.dims,
                     coords=agg.coords,
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

# TODO: add optional name parameter `name='binary'`
def binary(agg, values):
    return DataArray(_binary(agg.data, values),
                     name='binary',
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)

@ngjit
def _mean(data, excludes):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):

            exclude = False
            for ex in excludes:
                if data[y,x] == ex:
                    exclude = True
                    break

            if not exclude:
                a,b,c,d,e,f,g,h,i = [data[y-1, x-1], data[y, x-1], data[y+1, x-1],
                                     data[y-1, x],   data[y, x],   data[y+1, x],
                                     data[y-1, x+1], data[y, x+1], data[y+1, x+1]]
                out[y, x] = (a+b+c+d+e+f+g+h+i) / 9
            else:
                out[y, x] = data[y, x]
    return out

# TODO: add optional name parameter `name='mean'`
def mean(agg, passes=1, excludes=[np.nan]):
    """
    Returns Mean filtered array using a 3x3 window

    Parameters
    ----------
    agg : DataArray
    passes : int, number of times to run mean

    Returns
    -------
    data: DataArray
    """
    out = None
    for i in range(passes):
        if out is None:
            out = _mean(agg.data, tuple(excludes))
        else:
            out = _mean(out, tuple(excludes))

    return DataArray(out, name='mean', dims=agg.dims, coords=agg.coords, attrs=agg.attrs)


# TODO: add optional name parameter `name='terrain'`
def generate_terrain(canvas, seed=10, zfactor=4000, full_extent=None):
    """
    Generates a pseudo-random terrain which can be helpful for testing raster functions

    Parameters
    ----------
    canvas : ds.Canvas instance for passing output dimensions / ranges

    seed : seed for random number generator

    zfactor : used as multipler for z values

    full_extent : optional string, bbox<xmin, ymin, xmax, ymax>
      full extent of coordinate system.

    Returns
    -------
    terrain: DataArray

    Notes:
    ------
    Algorithm References:
     - This was inspired by Michael McHugh's 2016 PyCon Canada talk:
       https://www.youtube.com/watch?v=O33YV4ooHSo
     - https://www.redblobgames.com/maps/terrain-from-noise/
    """


    def _gen_heights(bumps):
        out = np.zeros(len(bumps))
        for i, b in enumerate(bumps):
            x = b[0]
            y = b[1]
            val = agg.data[y, x]
            if val >= 0.33 and val <= 3:
                out[i] = 0.1
        return out

    def _scale(value, old_range, new_range):
        return ((value - old_range[0]) / (old_range[1] - old_range[0])) * (new_range[1] - new_range[0]) + new_range[0]

    if not isinstance(canvas, Canvas):
        raise TypeError('canvas must be instance type datashader.Canvas')

    mercator_extent = (-np.pi * 6378137, -np.pi * 6378137, np.pi * 6378137, np.pi * 6378137)
    crs_extents = {'3857': mercator_extent}

    if isinstance(full_extent, str):
        full_extent = crs_extents[full_extent]

    elif full_extent is None:
        full_extent = (canvas.x_range[0], canvas.y_range[0], canvas.x_range[1], canvas.y_range[1])

    elif not isinstance(full_extent, (list, tuple)) and len(full_extent) != 4:
        raise TypeError('full_extent must be tuple(4) or str wkid')

    full_xrange = (full_extent[0], full_extent[2])
    full_yrange = (full_extent[1], full_extent[3])

    x_range_scaled = (_scale(canvas.x_range[0], full_xrange, (0.0, 1.0)),
                      _scale(canvas.x_range[1], full_xrange, (0.0, 1.0)))

    y_range_scaled = (_scale(canvas.y_range[0], full_yrange, (0.0, 1.0)),
                      _scale(canvas.y_range[1], full_yrange, (0.0, 1.0)))

    data = _gen_terrain(canvas.plot_width, canvas.plot_height, seed,
                        x_range=x_range_scaled, y_range=y_range_scaled)

    data = (data - np.min(data))/np.ptp(data)
    data[data < 0.3] = 0  # create water
    data *= zfactor

    # DataArray coords were coming back different from cvs.points...
    hack_agg = canvas.points(pd.DataFrame({'x': [],'y': []}), 'x', 'y')
    agg = DataArray(data,
                    name='terrain',
                    coords=hack_agg.coords,
                    dims=hack_agg.dims,
                    attrs={'res':1})

    return agg

def _gen_terrain(width, height, seed, x_range=None, y_range=None):

    if not x_range:
        x_range = (0, 1)

    if not y_range:
        y_range = (0, 1)

    # multiplier, (xfreq, yfreq)
    NOISE_LAYERS= ((1 / 2**i, (2**i, 2**i)) for i in range(16))

    linx = np.linspace(x_range[0], x_range[1], width, endpoint=False)
    liny = np.linspace(y_range[0], y_range[1], height, endpoint=False)
    x, y = np.meshgrid(linx, liny)

    height_map = None
    for i, (m, (xfreq, yfreq)) in enumerate(NOISE_LAYERS):
        noise = _perlin(x * xfreq, y * yfreq, seed=seed + i) * m
        if height_map is None:
            height_map = noise
        else:
            height_map += noise

    height_map /= (1.00 + 0.50 + 0.25 + 0.13 + 0.06 + 0.03)
    height_map = height_map ** 3
    return height_map


# TODO: change parameters to take agg instead of height / width
def bump(width, height, count=None, height_func=None, spread=1):
    """
    Generate a simple bump map

    Parameters
    ----------
    width : int
    height : int
    count : int (defaults: w * h / 10)
    height_func : function which takes x, y and returns a height value
    spread : tuple boundaries

    Returns
    -------
    bumpmap: DataArray

    Notes:
    ------
    Algorithm References:
     - http://www.mountaincartography.org/mt_hood/pdfs/nighbert_bump1.pdf
    """

    linx = range(width)
    liny = range(height)

    if count is None:
        count = width * height // 10

    if height_func is None:
        height_func = lambda bumps: np.ones(len(bumps))

    # create 2d array of random x, y for bump locations
    locs = np.empty((count, 2), dtype=np.uint16)
    locs[:, 0] = np.random.choice(linx, count)
    locs[:, 1] = np.random.choice(liny, count)

    heights = height_func(locs)

    bumps = _finish_bump(width, height, locs, heights, spread)
    return DataArray(bumps, dims=['y', 'x'], attrs=dict(res=1))

@ngjit
def _finish_bump(width, height, locs, heights, spread):
    out = np.zeros((height, width))
    rows, cols = out.shape
    s = spread ** 2  # removed sqrt for perf.
    for i in range(len(heights)):
        x = locs[i][0]
        y = locs[i][1]
        z = heights[i]
        out[y, x] = out[y, x] + z
        if s > 0:
            for nx in range(max(x - spread, 0), min(x + spread, width)):
                for ny in range(max(y - spread, 0), min(y + spread, height)):
                    d2 = (nx - x) * (nx - x) + (ny -  y) * (ny - y)
                    if d2 <= s:
                        out[ny, nx] = out[ny,nx] + (out[y, x] * (d2 / s))
    return out


# TODO: change parameters to take agg instead of height / width
def perlin(width, height, freq=(1, 1), seed=5):
    """
    Generate perlin noise aggregate

    Parameters
    ----------
    width : int
    height : int
    freq : tuple of (x, y) frequency multipliers
    seed : int

    Returns
    -------
    bumpmap: DataArray

    Notes:
    ------
    Algorithm References:
    - numba-ized from Paul Panzer example available here:
    https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy
     - http://www.mountaincartography.org/mt_hood/pdfs/nighbert_bump1.pdf
    """
    linx = range(width)
    liny = range(height)
    linx = np.linspace(0, 1, width, endpoint=False)
    liny = np.linspace(0, 1, height, endpoint=False)
    x, y = np.meshgrid(linx, liny)
    data = _perlin(x * freq[0], y * freq[1], seed=seed)
    data = (data - np.min(data))/np.ptp(data)
    return DataArray(data, dims=['y', 'x'], attrs=dict(res=1))


@ngjit
def _lerp(a, b, x):
    return a + x * (b-a)


@ngjit
def _fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


@ngjit
def _gradient(h, x, y):
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    dim_ = h.shape
    out = np.zeros(dim_)
    for j in range(dim_[1]):
        for i in range(dim_[0]):
            f = np.mod(h[i,j], 4)
            g = vectors[f]
            out[i,j] = g[0] * x[i,j] + g[1] * y[i,j]
    return out


def _perlin(x, y, seed=0):
    np.random.seed(seed)
    p = np.arange(2**20,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()

    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)

    # internal coordinates
    xf = x - xi
    yf = y - yi

    # fade factors
    u = _fade(xf)
    v = _fade(yf)

    # noise components
    n00 = _gradient(p[p[xi]+yi], xf, yf)
    n01 = _gradient(p[p[xi]+yi+1], xf, yf-1)
    n11 = _gradient(p[p[xi+1]+yi+1], xf-1, yf-1)
    n10 = _gradient(p[p[xi+1]+yi], xf-1, yf)

    # combine noises
    x1 = _lerp(n00, n10, u)
    x2 = _lerp(n01, n11, u)
    a = _lerp(x1, x2, v)
    return a
