from __future__ import absolute_import, division, print_function

from io import BytesIO

import numpy as np
import datashape
from dynd import nd
from PIL.Image import fromarray

from .colors import rgb
from .utils import is_missing, is_option


__all__ = ['Image', 'merge', 'stack', 'interpolate', 'colorize']


class Image(object):
    def __init__(self, img):
        self.img = img

    def _repr_png_(self):
        return self.to_pil()._repr_png_()

    def to_pil(self, origin='lower'):
        arr = self.img
        if origin == 'lower':
            arr = np.flipud(arr)
        return fromarray(arr, 'RGBA')

    def to_bytesio(self, format='png', origin='lower'):
        fp = BytesIO()
        self.to_pil(origin).save(fp, format)
        fp.seek(0)
        return fp


def _to_channels(data):
    return data.view([('r', 'uint8'), ('g', 'uint8'), ('b', 'uint8'),
                      ('a', 'uint8')])


def merge(*imgs):
    """Merge a number of images together, averaging the channels"""
    imgs = _to_channels(np.stack([i.img for i in imgs]))
    r = imgs['r'].mean(axis=0, dtype='f8').astype('uint8')
    g = imgs['g'].mean(axis=0, dtype='f8').astype('uint8')
    b = imgs['b'].mean(axis=0, dtype='f8').astype('uint8')
    a = imgs['a'].mean(axis=0, dtype='f8').astype('uint8')
    return Image(np.dstack([r, g, b, a]).view(np.uint32).reshape(a.shape))


def stack(*imgs):
    """Merge a number of images together, overlapping earlier images with
    later ones."""
    out = imgs[0].img.copy()
    for img in imgs[1:]:
        out = np.where(_to_channels(img.img)['a'] == 0, out, img.img)
    return Image(out)


_interpolate_lookup = {'log': np.log1p,
                       'cbrt': lambda x: x ** (1/3.),
                       'linear': lambda x: x}


def _normalize_interpolate_how(how):
    if callable(how):
        return how
    elif how in _interpolate_lookup:
        return _interpolate_lookup[how]
    raise ValueError("Unknown interpolation method: {0}".format(how))


def interpolate(agg, low, high, how='log'):
    """Convert an aggregate to an image.

    Parameters
    ----------
    agg : A dynd array
    low : color name or tuple
        The color for the low end of the scale. Can be specified either by
        name, hexcode, or as a tuple of ``(red, green, blue)`` values.
    high : color name or tuple
        The color for the high end of the scale
    how : string or callable
        The interpolation method to use. Valid strings are 'log' [default],
        'cbrt', and 'linear'. Callables take a 2-dimensional array of
        magnitudes at each pixel, and should return a numeric array of the same
        shape.
    """
    if is_option(agg.dtype):
        buffer = nd.as_numpy(agg.view_scalars(agg.dtype.value_type))
        missing = is_missing(buffer)
    else:
        buffer = nd.as_numpy(agg)
        missing = (buffer == 0)
    offset = buffer[~missing].min()
    data = _normalize_interpolate_how(how)(buffer + offset)
    span = [data[~missing].min(), data[~missing].max()]
    rspan, gspan, bspan = zip(rgb(low), rgb(high))
    r = np.interp(data, span, rspan, left=255).astype(np.uint8)
    g = np.interp(data, span, gspan, left=255).astype(np.uint8)
    b = np.interp(data, span, bspan, left=255).astype(np.uint8)
    a = np.full_like(r, 255)
    img = np.dstack([r, g, b, a]).view(np.uint32).reshape(a.shape)
    return Image(np.where(missing, 0, img))


def colorize(agg, color_key, how='log', min_alpha=20):
    """Color a record aggregate by field.

    Parameters
    ----------
    agg : dynd array
    color_key : dict or iterable
        A mapping of fields to colors. Can be either a ``dict`` mapping from
        field name to colors, or an iterable of colors in the same order as the
        record fields.
    how : string or callable
        The interpolation method to use. Valid strings are 'log' [default],
        'cbrt', and 'linear'. Callables take a 2-dimensional array of
        magnitudes at each pixel, and should return a numeric array of the same
        shape.
    min_alpha : float, optional
        The minimum alpha value to use for non-empty pixels, in [0, 255].
    """
    agg_dshape = datashape.dshape(agg.dtype.dshape)
    if not datashape.isrecord(agg_dshape):
        raise ValueError("Datashape of aggregate must be record type")
    if not isinstance(color_key, dict):
        color_key = dict(zip(agg_dshape.measure.names, color_key))
    if len(color_key) != len(agg_dshape.measure.names):
        raise ValueError("Number of colors doesn't match number of fields")
    if not (0 <= min_alpha <= 255):
        raise ValueError("min_alpha must be between 0 and 255")
    colors = [rgb(color_key[c]) for c in agg_dshape.measure.names]
    rs, gs, bs = map(np.array, zip(*colors))
    data = np.dstack([nd.as_numpy(getattr(agg, f)).astype('f8') for f in
                      agg_dshape.measure.names])
    total = data.sum(axis=2)
    r = (data.dot(rs)/total).astype(np.uint8)
    g = (data.dot(gs)/total).astype(np.uint8)
    b = (data.dot(bs)/total).astype(np.uint8)
    a = _normalize_interpolate_how(how)(total)
    a = ((255 - min_alpha) * a/a.max() + min_alpha).astype(np.uint8)
    white = (total == 0)
    r[white] = g[white] = b[white] = 255
    a[white] = 0
    return Image(np.dstack([r, g, b, a]).view(np.uint32).reshape(a.shape))
