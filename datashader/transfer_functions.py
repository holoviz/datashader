from __future__ import absolute_import, division, print_function

from io import BytesIO

import numpy as np
import numba as nb
import xarray as xr
from PIL.Image import fromarray
from toolz import memoize

from .colors import rgb
from .composite import composite_op_lookup


__all__ = ['Image', 'merge', 'stack', 'interpolate', 'colorize', 'spread']


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


def spread(img, px=1, shape='circle', how='over'):
    """Spread pixels in an image.

    Spreading expands each pixel a certain number of pixels on all sides
    according to a given shape, merging pixels using a specified compositing
    operator. This can be useful to make sparse plots more visible.

    Parameters
    ----------
    img : Image
    px : int, optional
        Number of pixels to spread on all sides
    shape : str, optional
        The shape to spread by. Options are 'circle' [default] or 'square'.
    how : str, optional
        The name of the compositing operator to use when combining pixels.
    """
    if not isinstance(img, Image):
        raise TypeError("Expected `Image`, got: `{0}`".format(type(img)))
    elif not isinstance(px, int) or px < 0:
        raise ValueError("``px`` must be an integer >= 0")
    if px == 0:
        return img
    mask = _mask_lookup[shape](px)
    kernel = _build_spread_kernel(how)
    w = mask.shape[0]
    extra = w // 2
    M, N = img.shape
    buf = np.zeros((M + 2*extra, N + 2*extra), dtype='uint32')
    kernel(img.data, mask, buf)
    out = buf[extra:-extra, extra:-extra].copy()
    return Image(out, dims=img.dims, coords=img.coords)


@memoize
def _build_spread_kernel(how):
    """Build a spreading kernel for a given composite operator"""
    op = composite_op_lookup[how]

    @nb.jit(nopython=True, nogil=True)
    def kernel(arr, mask, out):
        M, N = arr.shape
        w = mask.shape[0]
        for y in range(M):
            for x in range(N):
                el = arr[y, x]
                # Skip if data is transparent
                if (el >> 24) & 255:
                    for i in range(w):
                        for j in range(w):
                            # Skip if mask is False at this value
                            if mask[i, j]:
                                out[i + y, j + x] = op(el, out[i + y, j + x])
    return kernel


def _square_mask(px):
    """Produce a square mask with sides of length ``2 * px + 1``"""
    px = int(px)
    w = 2 * px + 1
    return np.ones((w, w), dtype='bool')


def _circle_mask(r):
    """Produce a circular mask with a diameter of ``2 * r + 1``"""
    r = int(r)
    w = 2 * r + 1
    mask = np.zeros((w, w), dtype='bool')
    _fill_circle(mask, r, r, r, True)
    return mask


_mask_lookup = {'square': _square_mask,
                'circle': _circle_mask}


@nb.jit(nopython=True, nogil=True)
def _fill_circle(arr, x0, y0, r, val):
    """General circle filling routine.

    Low level, no bounds checking performed. Has potential to segfault if
    misused.

    Parameters
    ----------
    arr : 2d array
    x0, y0 : int
        Center coordinates of the circle.
    r : int
        Circle radius in pixels. Note that this excludes the center pixel - the
        resulting circle will be ``2 * r + 1`` pixels in diameter.
    val : scalar
        Fill value
    """
    x = r
    y = 0
    do2 = 1 - x

    # Special cases for r == 0 (fastpath) and r == 1. The normal algorithm will
    # draw a 3x3 grid for this case, but a "+" looks a bit better in my opinion
    if r == 0:
        arr[y0, x0] = 1
        return
    elif r == 1:
        _fill_span(arr, y0, x0 - 1, x0 + 1, val)
        arr[y0 + 1, x0] = arr[y0 - 1, x0] = val
        return
    # Determine width
    while y <= x:
        y += 1
        if do2 <= 0:
            do2 += 2 * y + 1
        else:
            x -= 1
            do2 += 2 * (y - x) + 1
            break
    width = y - 1
    # Fill in spans
    _fill_span(arr, y0 + r, x0 - width, x0 + width, val)
    _fill_span(arr, y0 - r, x0 - width, x0 + width, val)
    for row in range(-width, width + 1):
        _fill_span(arr, y0 + row, x0 - r, x0 + r, val)
    # Fill in remainder
    while y < r:
        _fill_span(arr, y0 + y, x0 - x, x0 + x, val)
        _fill_span(arr, y0 - y, x0 - x, x0 + x, val)
        y += 1
        if do2 <= 0:
            do2 += 2 * y + 1
        else:
            x -= 1
            do2 += 2 * (y - x) + 1


@nb.jit(nopython=True, nogil=True)
def _fill_span(arr, y, l, r, val):
    """Fill a horizontal span with ``val``"""
    for i in range(l, r + 1):
        arr[y, i] = val
