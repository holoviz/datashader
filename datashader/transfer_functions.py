from __future__ import absolute_import, division, print_function

from io import BytesIO

import numpy as np
import toolz as tz
import xarray as xr
from PIL.Image import fromarray


from .colors import rgb
from .composite import composite_op_lookup, source
from .utils import ngjit


__all__ = ['Image', 'stack', 'interpolate', 'colorize', 'set_background',
           'spread']


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


def stack(*imgs, **kwargs):
    """Combine images together, overlaying later images onto earlier ones.

    Parameters
    ----------
    imgs : iterable of Image
        The images to combine.
    how : str, optional
        The compositing operator to combine pixels. Default is `'over'`.
    """
    if not imgs:
        raise ValueError("No images passed in")
    for i in imgs:
        if not isinstance(i, Image):
            raise TypeError("Expected `Image`, got: `{0}`".format(type(i)))
    op = composite_op_lookup[kwargs.get('how', 'over')]
    if len(imgs) == 1:
        return imgs[0]
    imgs = xr.align(*imgs, copy=False, join='outer')
    out = tz.reduce(tz.flip(op), [i.data for i in imgs])
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


def interpolate(agg, low="lightblue", high="darkblue", how='cbrt', cmap=None):    
    """Convert a 2D DataArray to an image.

    Parameters
    ----------
    agg : DataArray
    low, high : color name or tuple, optional
        The color for the low and high ends of the scale. Can be specified
        either by name, hexcode, or as a tuple of ``(red, green, blue)``
        values.
    how : str or callable, optional
        The interpolation method to use. Valid strings are 'cbrt' [default],
        'log', and 'linear'. Callables take a 2-dimensional array of
        magnitudes at each pixel, and should return a numeric array of the same
        shape.
    cmap : str, optional
        If specified, interpolate along the named colormap, as provided by
        matplotlib.  Overrides low,high.
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

    if cmap is None:
        rspan, gspan, bspan = zip(rgb(low), rgb(high))
        r = np.interp(data, span, rspan, left=255).astype(np.uint8)
        g = np.interp(data, span, gspan, left=255).astype(np.uint8)
        b = np.interp(data, span, bspan, left=255).astype(np.uint8)
        a = np.where(np.isnan(data), 0, 255).astype(np.uint8)
        img = np.dstack([r, g, b, a]).view(np.uint32).reshape(a.shape)
    else:
        from matplotlib.cm import get_cmap
        mapper = get_cmap(cmap)
        tmp = mapper(data, bytes=True)
        tmp[:,:,3] = np.where(np.isnan(data), 0, 255).astype(np.uint8)
        img = tmp.view(np.uint32).reshape(data.shape)
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
    how : str or callable, optional
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


def set_background(img, color=None):
    """Return a new image, with the background set to `color`.

    Parameters
    -----------------
    img : Image
    color : color name or tuple, optional
        The background color. Can be specified either by name, hexcode, or as a
        tuple of ``(red, green, blue)`` values.
    """
    if not isinstance(img, Image):
        raise TypeError("Expected `Image`, got: `{0}`".format(type(img)))
    if color is None:
        return img
    background = np.uint8(rgb(color) + (255,)).view('uint32')[0]
    data = source(img.data, background)
    return Image(data, coords=img.coords, dims=img.dims)


def spread(img, px=1, shape='circle', how='over', mask=None):
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
    mask : ndarray, shape (M, M), optional
        The mask to spread over. If provided, this mask is used instead of
        generating one based on `px` and `shape`. Must be a square array
        with odd dimensions. Pixels are spread from the center of the mask to
        locations where the mask is True.
    """
    if not isinstance(img, Image):
        raise TypeError("Expected `Image`, got: `{0}`".format(type(img)))
    if mask is None:
        if not isinstance(px, int) or px < 0:
            raise ValueError("``px`` must be an integer >= 0")
        if px == 0:
            return img
        mask = _mask_lookup[shape](px)
    elif not (isinstance(mask, np.ndarray) and mask.ndim == 2 and
              mask.shape[0] == mask.shape[1] and mask.shape[0] % 2 == 1):
        raise ValueError("mask must be a square 2 dimensional ndarray with "
                         "odd dimensions.")
        mask = mask if mask.dtype == 'bool' else mask.astype('bool')
    kernel = _build_spread_kernel(how)
    w = mask.shape[0]
    extra = w // 2
    M, N = img.shape
    buf = np.zeros((M + 2*extra, N + 2*extra), dtype='uint32')
    kernel(img.data, mask, buf)
    out = buf[extra:-extra, extra:-extra].copy()
    return Image(out, dims=img.dims, coords=img.coords)


@tz.memoize
def _build_spread_kernel(how):
    """Build a spreading kernel for a given composite operator"""
    op = composite_op_lookup[how]

    @ngjit
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
    x = np.arange(-r, r + 1, dtype='i4')
    bound = r + 0.5 if r > 1 else r
    return np.where(np.sqrt(x**2 + x[:, None]**2) <= bound, True, False)


_mask_lookup = {'square': _square_mask,
                'circle': _circle_mask}
