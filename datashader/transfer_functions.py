from __future__ import absolute_import, division, print_function

import warnings
from collections import Iterator
from io import BytesIO

import numpy as np
import numba as nb
import toolz as tz
import xarray as xr
from PIL.Image import fromarray


from .colors import rgb
from .composite import composite_op_lookup, over
from .utils import ngjit


__all__ = ['Image', 'stack', 'interpolate', 'colorize', 'set_background',
           'spread', 'dynspread']


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


def eq_hist(data, mask=None, nbins=256*256):
    """Return a numpy array after histogram equalization.

    For use in `interpolate`.

    Parameters
    ----------
    data : ndarray
    mask : ndarray, optional
       Boolean array of missing points. Where True, the output will be `NaN`.
    nbins : int, optional
        Number of bins to use. Note that this argument is ignored for integer
        arrays, which bin by the integer values directly.

    Notes
    -----
    This function is adapted from the implementation in scikit-image [1]_.

    References
    ----------
    .. [1] http://scikit-image.org/docs/stable/api/skimage.exposure.html#equalize-hist
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be np.ndarray")
    data2 = data if mask is None else data[~mask]
    if np.issubdtype(data2.dtype, np.integer):
        hist = np.bincount(data2.ravel())
        bin_centers = np.arange(len(hist))
        idx = np.nonzero(hist)[0][0]
        hist, bin_centers = hist[idx:], bin_centers[idx:]
    else:
        hist, bin_edges = np.histogram(data2, bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cdf = hist.cumsum()
    cdf = cdf / float(cdf[-1])
    out = np.interp(data.flat, bin_centers, cdf).reshape(data.shape)
    return out if mask is None else np.where(mask, np.nan, out)


_interpolate_lookup = {'log': lambda d, m: np.log1p(np.where(m, np.nan, d)),
                       'cbrt': lambda d, m: np.where(m, np.nan, d)**(1/3.),
                       'linear': lambda d, m: np.where(m, np.nan, d),
                       'eq_hist': eq_hist}


def _normalize_interpolate_how(how):
    if callable(how):
        return how
    elif how in _interpolate_lookup:
        return _interpolate_lookup[how]
    raise ValueError("Unknown interpolation method: {0}".format(how))


def interpolate(agg, low=None, high=None, cmap=None, how='eq_hist'):
    """Convert a 2D DataArray to an image.

    Data is converted to an image either by interpolating between a `low` and
    `high` color [default], or by a specified colormap.

    Parameters
    ----------
    agg : DataArray
    low, high : color name or tuple, optional
        Deprecated. The color for the low and high ends of the scale. Can be
        specified either by name, hexcode, or as a tuple of ``(red, green,
        blue)`` values.
    cmap : list of colors or matplotlib.colors.Colormap, optional
        The colormap to use. Can be either a list of colors (in any of the
        formats described above), or a matplotlib colormap object.
        Default is `["lightblue", "darkblue"]`
    how : str or callable, optional
        The interpolation method to use. Valid strings are 'cbrt' [default],
        'log', 'linear', and 'eq_hist'. Callables take 2 arguments - a
        2-dimensional array of magnitudes at each pixel, and a boolean mask
        array indicating missingness. They should return a numeric array of the
        same shape, with `NaN`s where the mask was True.
    """
    if not isinstance(agg, xr.DataArray):
        raise TypeError("agg must be instance of DataArray")
    if agg.ndim != 2:
        raise ValueError("agg must be 2D")
    if cmap is not None:
        if low or high:
            raise ValueError("Can't provide both `cmap` and `low` or `high`")
    else:
        # Defaults
        cmap = ['lightblue', 'darkblue']
        if low or high:
            with warnings.catch_warnings():
                warnings.simplefilter('always', DeprecationWarning)
                w = DeprecationWarning("Using `low` and `high` is deprecated. "
                                       "Instead use `cmap=[low, high]`")
                warnings.warn(w)
            cmap = [low or cmap[0], high or cmap[1]]
    how = _normalize_interpolate_how(how)
    data = agg.data
    if np.issubdtype(data.dtype, np.bool_):
        mask = ~data
        interp = data
    else:
        if np.issubdtype(data.dtype, np.integer):
            mask = data == 0
        else:
            mask = np.isnan(data)
        offset = data[~mask].min()
        interp = data - offset
    data = how(interp, mask)
    span = [np.nanmin(data), np.nanmax(data)]
    if isinstance(cmap, Iterator):
        cmap = list(cmap)
    if isinstance(cmap, list):
        rspan, gspan, bspan = np.array(list(zip(*map(rgb, cmap))))
        span = np.linspace(span[0], span[1], len(cmap))
        r = np.interp(data, span, rspan, left=255).astype(np.uint8)
        g = np.interp(data, span, gspan, left=255).astype(np.uint8)
        b = np.interp(data, span, bspan, left=255).astype(np.uint8)
        a = np.where(np.isnan(data), 0, 255).astype(np.uint8)
        img = np.dstack([r, g, b, a]).view(np.uint32).reshape(a.shape)
    elif callable(cmap):
        # Assume callable is matplotlib colormap
        img = cmap((data - span[0])/(span[1] - span[0]), bytes=True)
        img[:, :, 3] = np.where(np.isnan(data), 0, 255).astype(np.uint8)
        img = img.view(np.uint32).reshape(data.shape)
    else:
        raise TypeError("Expected `cmap` of `matplotlib.colors.Colormap` or "
                        "`list`, got: '{0}'".format(type(cmap)))
    return Image(img, coords=agg.coords, dims=agg.dims)


def colorize(agg, color_key, how='eq_hist', min_alpha=20):
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
        'log', and 'linear'. Callables take 2 arguments - a 2-dimensional
        array of magnitudes at each pixel, and a boolean mask array indicating
        missingness. They should return a numeric array of the same shape, with
        `NaN`s where the mask was True.
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
    offset = total.min()
    mask = np.isnan(total)
    if offset == 0:
        mask = mask | (total <= 0)
        offset = total[total > 0].min()
    a = _normalize_interpolate_how(how)(total - offset, mask)
    a = np.interp(a, [np.nanmin(a), np.nanmax(a)],
                  [min_alpha, 255], left=0, right=255).astype(np.uint8)
    r[mask] = g[mask] = b[mask] = 255
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
    data = over(img.data, background)
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
    return np.where(np.sqrt(x**2 + x[:, None]**2) <= r+0.5, True, False)


_mask_lookup = {'square': _square_mask,
                'circle': _circle_mask}


def dynspread(img, threshold=0.5, max_px=3, shape='circle', how='over'):
    """Spread pixels in an image dynamically based on the image density.

    Spreading expands each pixel a certain number of pixels on all sides
    according to a given shape, merging pixels using a specified compositing
    operator. This can be useful to make sparse plots more visible. Dynamic
    spreading determines how many pixels to spread based on a density
    heuristic.

    Parameters
    ----------
    img : Image
    threshold : float, optional
        A tuning parameter in [0, 1]. Indicates the minimum value for the
        density heuristic.
    max_px : int, optional
        Maximum number of pixels to spread on all sides.
    shape : str, optional
        The shape to spread by. Options are 'circle' [default] or 'square'.
    how : str, optional
        The name of the compositing operator to use when combining pixels.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if not isinstance(max_px, int) or max_px < 0:
        raise ValueError("max_px must be >= 0")
    # Simple linear search. Not super efficient, but max_px is usually small.
    for px in range(max_px + 1):
        out = spread(img, px, shape=shape, how=how)
        if _density(out.data) >= threshold:
            break
    return out


@nb.jit(nopython=True, nogil=True, cache=True)
def _density(arr):
    """Compute a density heuristic of an image.

    The density is a number in [0, 1], and indicates the normalized mean number
    of non-empty adjacent pixels for each non-empty pixel.
    """
    M, N = arr.shape
    cnt = total = 0
    for y in range(1, M - 1):
        for x in range(1, N - 1):
            if (arr[y, x] >> 24) & 255:
                cnt += 1
                for i in range(y - 1, y + 2):
                    for j in range(x - 1, x + 2):
                        if (arr[i, j] >> 24) & 255:
                            total += 1
    return (total - cnt)/(cnt * 8) if cnt else np.inf
