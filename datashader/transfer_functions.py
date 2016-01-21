from __future__ import absolute_import, division, print_function

from io import BytesIO

import numpy as np
from dynd import nd
from PIL.Image import fromarray

from .colors import rgb
from .utils import is_missing, is_option


__all__ = ['Image', 'merge', 'stack', 'interpolate']


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
    how : string
        The interpolation method to use. Options are 'log' [default], 'cbrt',
        and 'linear'.
    """
    if how == 'log':
        f = np.log1p
    elif how == 'cbrt':
        f = lambda x: x ** (1/3.)
    elif how == 'linear':
        f = lambda x: x
    else:
        raise ValueError("Unknown interpolation method: {0}".format(how))
    if is_option(agg.dtype):
        buffer = nd.as_numpy(agg.view_scalars(agg.dtype.value_type))
        missing = is_missing(buffer)
    else:
        buffer = nd.as_numpy(agg)
        missing = (buffer == 0)
    offset = buffer[~missing].min()

    data = f(buffer + offset)
    span = [data[~missing].min(), data[~missing].max()]
    rspan, gspan, bspan = zip(rgb(low), rgb(high))
    r = np.interp(data, span, rspan, left=255).astype(np.uint8)
    g = np.interp(data, span, gspan, left=255).astype(np.uint8)
    b = np.interp(data, span, bspan, left=255).astype(np.uint8)
    a = np.full_like(r, 255)
    img = np.dstack([r, g, b, a]).view(np.uint32).reshape(a.shape)
    return Image(np.where(missing, 0, img))
