from __future__ import absolute_import, division, print_function

from io import BytesIO

import numpy as np
from dynd import nd
from PIL.Image import fromarray

from .aggregates import ScalarAggregate, CategoricalAggregate, _validate_axis
from .colors import rgb
from .utils import dynd_to_np_mask


__all__ = ['Image', 'merge', 'stack', 'interpolate', 'colorize']


class Image(object):
    def __init__(self, img, x_axis=None, y_axis=None):
        self.img = img
        self.x_axis = _validate_axis(x_axis)
        self.y_axis = _validate_axis(y_axis)

    @property
    def shape(self):
        return self.img.shape

    def __repr__(self):
        return 'Image<shape={0}>'.format(self.shape)

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


def _validate_images(imgs):
    if not imgs:
        raise ValueError("No images passed in")
    for i in imgs:
        if not isinstance(i, Image):
            raise TypeError("Expected `Image`, got: `{0}`".format(type(i)))
    shapes = set(i.shape for i in imgs)
    x_axis = set(i.x_axis for i in imgs)
    y_axis = set(i.y_axis for i in imgs)
    if len(shapes) > 1 or len(x_axis) > 1 or len(y_axis) > 1:
        raise NotImplementedError("Operations between images with "
                                  "non-matching axis or shape")


def merge(*imgs):
    """Merge a number of images together, averaging the channels"""
    _validate_images(imgs)
    if len(imgs) == 1:
        return imgs[0]
    x_axis, y_axis = imgs[0].x_axis, imgs[0].y_axis
    imgs = _to_channels(np.stack([i.img for i in imgs]))
    r = imgs['r'].mean(axis=0, dtype='f8').astype('uint8')
    g = imgs['g'].mean(axis=0, dtype='f8').astype('uint8')
    b = imgs['b'].mean(axis=0, dtype='f8').astype('uint8')
    a = imgs['a'].mean(axis=0, dtype='f8').astype('uint8')
    return Image(np.dstack([r, g, b, a]).view(np.uint32).reshape(a.shape),
                 x_axis, y_axis)


def stack(*imgs):
    """Merge a number of images together, overlapping earlier images with
    later ones."""
    _validate_images(imgs)
    if len(imgs) == 1:
        return imgs[0]
    out = imgs[0].img.copy()
    for img in imgs[1:]:
        out = np.where(_to_channels(img.img)['a'] == 0, out, img.img)
    return Image(out, imgs[0].x_axis, imgs[0].y_axis)


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
    """Convert a ScalarAggregate to an image.

    Parameters
    ----------
    agg : ScalarAggregate
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
    if not isinstance(agg, ScalarAggregate):
        raise TypeError("agg must be instance of ScalarAggregate")
    buffer, missing = dynd_to_np_mask(agg._data)
    offset = buffer[~missing].min()
    data = _normalize_interpolate_how(how)(buffer + offset)
    span = [data[~missing].min(), data[~missing].max()]
    rspan, gspan, bspan = zip(rgb(low), rgb(high))
    r = np.interp(data, span, rspan, left=255).astype(np.uint8)
    g = np.interp(data, span, gspan, left=255).astype(np.uint8)
    b = np.interp(data, span, bspan, left=255).astype(np.uint8)
    a = np.full_like(r, 255)
    img = np.dstack([r, g, b, a]).view(np.uint32).reshape(a.shape)
    return Image(np.where(missing, 0, img),
                 agg.x_axis, agg.y_axis)


def colorize(agg, color_key, how='log', min_alpha=20):
    """Color a CategoricalAggregate by field.

    Parameters
    ----------
    agg : CategoricalAggregate
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
    if not isinstance(agg, CategoricalAggregate):
        raise TypeError("agg must be instance of CategoricalAggregate")
    if not isinstance(color_key, dict):
        color_key = dict(zip(agg.cats, color_key))
    if len(color_key) != len(agg.cats):
        raise ValueError("Number of colors doesn't match number of fields")
    if not (0 <= min_alpha <= 255):
        raise ValueError("min_alpha must be between 0 and 255")
    colors = [rgb(color_key[c]) for c in agg.cats]
    rs, gs, bs = map(np.array, zip(*colors))
    data = nd.as_numpy(agg._data).astype('f8')
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
                 agg.x_axis, agg.y_axis)
