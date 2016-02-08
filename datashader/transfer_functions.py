from __future__ import absolute_import, division, print_function

from io import BytesIO

import numpy as np
import xarray as xr
from PIL.Image import fromarray

from .colors import rgb


__all__ = ['Image', 'merge', 'stack', 'interpolate', 'colorize']


class Image(xr.DataArray):
    __array_priority__ = 70

    def _repr_png_(self):
        return self.to_pil()._repr_png_()

    def to_pil(self, origin='lower'):
        arr = np.flipud(self.data) if origin == 'lower' else self.data
        return fromarray(arr, 'RGBA')

    def to_bytesio(self, format='png', origin='lower'):
        fp = BytesIO()
        self.to_pil(origin).save(fp, format)
        fp.seek(0)
        return fp


def _to_channels(data):
    return data.view([('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')])


def _validate_images(imgs):
    if not imgs:
        raise ValueError("No images passed in")
    for i in imgs:
        if not isinstance(i, Image):
            raise TypeError("Expected `Image`, got: `{0}`".format(type(i)))


def merge(*imgs):
    """Merge a number of images together, averaging the channels"""
    _validate_images(imgs)
    if len(imgs) == 1:
        return imgs[0]
    imgs = xr.align(*imgs, copy=False, join='outer')
    coords, dims = imgs[0].coords, imgs[0].dims
    imgs = _to_channels(np.stack([i.data for i in imgs]))
    r = imgs['r'].mean(axis=0, dtype='f8').astype('uint8')
    g = imgs['g'].mean(axis=0, dtype='f8').astype('uint8')
    b = imgs['b'].mean(axis=0, dtype='f8').astype('uint8')
    a = imgs['a'].mean(axis=0, dtype='f8').astype('uint8')
    out = np.dstack([r, g, b, a]).view(np.uint32).reshape(a.shape)
    return Image(out, coords=coords, dims=dims)


def stack(*imgs):
    """Merge a number of images together, overlapping earlier images with
    later ones."""
    _validate_images(imgs)
    if len(imgs) == 1:
        return imgs[0]
    imgs = xr.align(*imgs, copy=False, join='outer')
    out = imgs[0].data.copy()
    for img in imgs[1:]:
        out = np.where(_to_channels(img.data)['a'] == 0, out, img.data)
    return Image(out, coords=imgs[0].coords, dims=imgs[0].dims)


_interpolate_lookup = {'log': np.log1p,
                       'cbrt': lambda x: x ** (1/3.),
                       'linear': lambda x: x}


def _normalize_interpolate_how(how):
    if callable(how):
        return how
    elif how in _interpolate_lookup:
        return _interpolate_lookup[how]
    raise ValueError("Unknown interpolation method: {0}".format(how))


def interpolate(agg, low="lightblue", high="darkblue", how='cbrt'):
    """Convert a 2D DataArray to an image.

    Parameters
    ----------
    agg : DataArray
    low : color name or tuple
        The color for the low end of the scale. Can be specified either by
        name, hexcode, or as a tuple of ``(red, green, blue)`` values.
    high : color name or tuple
        The color for the high end of the scale
    how : string or callable
        The interpolation method to use. Valid strings are 'cbrt' [default],
        'log', and 'linear'. Callables take a 2-dimensional array of
        magnitudes at each pixel, and should return a numeric array of the same
        shape.
    """
    if not isinstance(agg, xr.DataArray):
        raise TypeError("agg must be instance of DataArray")
    if agg.ndim != 2:
        raise ValueError("agg must be 2D")
    offset = agg.min()
    if offset == 0:
        agg = agg.where(agg > 0)
        offset = agg.min()
    how = _normalize_interpolate_how(how)
    data = how(agg - offset)
    span = [np.nanmin(data), np.nanmax(data)]
    rspan, gspan, bspan = zip(rgb(low), rgb(high))
    r = np.interp(data, span, rspan, left=255).astype(np.uint8)
    g = np.interp(data, span, gspan, left=255).astype(np.uint8)
    b = np.interp(data, span, bspan, left=255).astype(np.uint8)
    a = np.where(np.isnan(data), 0, 255).astype(np.uint8)
    img = np.dstack([r, g, b, a]).view(np.uint32).reshape(a.shape)
    return Image(img, coords=agg.coords, dims=agg.dims)


def colorize(agg, color_key, how='cbrt', min_alpha=20):
    """Color a CategoricalAggregate by field.

    Parameters
    ----------
    agg : DataArray
    color_key : dict or iterable
        A mapping of fields to colors. Can be either a ``dict`` mapping from
        field name to colors, or an iterable of colors in the same order as the
        record fields.
    how : string or callable
        The interpolation method to use. Valid strings are 'cbrt' [default],
        'log', and 'linear'. Callables take a 2-dimensional array of
        magnitudes at each pixel, and should return a numeric array of the same
        shape.
    min_alpha : float, optional
        The minimum alpha value to use for non-empty pixels, in [0, 255].
    """
    if not isinstance(agg, xr.DataArray):
        raise TypeError("agg must be an instance of DataArray")
    if not agg.ndim == 3:
        raise ValueError("agg must be 3D")
    cats = agg.indexes[agg.dims[-1]]
    if not isinstance(color_key, dict):
        color_key = dict(zip(cats, color_key))
    if len(color_key) != len(cats):
        raise ValueError("Number of colors doesn't match number of fields")
    if not (0 <= min_alpha <= 255):
        raise ValueError("min_alpha must be between 0 and 255")
    colors = [rgb(color_key[c]) for c in cats]
    rs, gs, bs = map(np.array, zip(*colors))
    data = agg.data
    total = data.sum(axis=2)
    r = (data.dot(rs)/total).astype(np.uint8)
    g = (data.dot(gs)/total).astype(np.uint8)
    b = (data.dot(bs)/total).astype(np.uint8)
    a = _normalize_interpolate_how(how)(total)
    a = ((255 - min_alpha) * a/a.max() + min_alpha).astype(np.uint8)
    white = (total == 0)
    r[white] = g[white] = b[white] = 255
    a[white] = 0
    return Image(np.dstack([r, g, b, a]).view(np.uint32).reshape(a.shape),
                 dims=agg.dims[:-1], coords=list(agg.coords.values())[:-1])
