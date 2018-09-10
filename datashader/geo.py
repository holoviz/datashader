"""
This module contains geoscience-related transfer functions whose use is completely optional.
"""

from __future__ import division

import numpy as np
import datashader.transfer_functions as tf

from datashader.colors import rgb
from datashader.utils import ngjit
from xarray import DataArray

__all__ = ['mean', 'binary', 'slope', 'aspect', 'ndvi', 'hillshade']


def _shade(altituderad, aspect, azimuthrad, slope):
    shade = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) * np.cos(azimuthrad - aspect)
    return shade

def _threshold_hs(img):
    dt = np.dtype((np.int32, {'r': (np.uint8, 0),
                              'g': (np.uint8, 1),
                              'b': (np.uint8, 2),
                              'a': (np.uint8, 3)}))
    img.data = np.where(img.data.view(dtype=dt)['r'] > 105, 0, img) # awk.
    return img

def _simple_hs(altitude, aspect, azimuth, slope, cmap, alpha, out_type='image'):
    _shaded = _shade(altitude, aspect, azimuth, slope)

    agg = DataArray(_shaded, dims=['y','x'])
    agg.data = np.where(agg > agg.mean(), 0, agg)
    if out_type == 'image':
        img = tf.shade(agg, cmap=cmap, how='linear', alpha=alpha)
        return _threshold_hs(img)
    elif out_type == 'data':
        return agg
    else:
        raise ValueError("Unknown  out_type: {0}".format(out_type))


def _mdow_hs(altitude, aspect, azimuth, slope, cmap, alpha, out_type='image'):
    alt = np.deg2rad(30)
    shade = np.sum([_shade(alt, aspect, np.deg2rad(225), slope),
                    _shade(alt, aspect, np.deg2rad(270), slope),
                    _shade(alt, aspect, np.deg2rad(315), slope),
                    _shade(alt, aspect, np.deg2rad(360), slope)], axis=0)
    shade /= 4

    agg = DataArray(shade, dims=['y', 'x'])
    agg.data = np.where(agg > agg.mean(), 0, agg)
    if out_type == 'image':
        img = tf.shade(agg, cmap=cmap, how='linear', alpha=alpha)
        return _threshold_hs(img)
    elif out_type == 'data':
        return agg
    else:
        raise ValueError("Unknown  out_type: {0}".format(out_type))


_hillshade_lookup = {'simple': _simple_hs,
                     'mdow': _mdow_hs}

def _normalize_hillshade_how(how):
    if how in _hillshade_lookup:
        return _hillshade_lookup[how]
    raise ValueError("Unknown hillshade method: {0}".format(how))

def hillshade(agg,
              altitude=30,
              azimuth=315,
              alpha=70,
              how='mdow',
              out_type='image',
              cmap=['#C7C7C7', '#000000']):
    """Convert a 2D DataArray to an hillshaded image with specified colormap.

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
    global _threshold_hs

    if not isinstance(agg, DataArray):
        raise TypeError("agg must be instance of DataArray")

    azimuthrad = np.deg2rad(azimuth)
    altituderad = np.deg2rad(altitude)
    y, x = np.gradient(agg.data)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-y, x)
    how = _normalize_hillshade_how(how)
    return how(altituderad, aspect, azimuthrad, slope, cmap, alpha, out_type)


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


def mean(agg, pad=None):
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
