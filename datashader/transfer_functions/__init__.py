from __future__ import annotations

from collections.abc import Iterator

from io import BytesIO

import warnings

import numpy as np
import numba as nb
import toolz as tz
import xarray as xr
import dask.array as da
from PIL.Image import fromarray

from datashader.colors import rgb, Sets1to3
from datashader.utils import nansum_missing, ngjit

try:
    import cupy
except Exception:
    cupy = None

__all__ = ['Image', 'stack', 'shade', 'set_background', 'spread', 'dynspread']


class Image(xr.DataArray):
    __slots__ = ()
    __array_priority__ = 70
    border=1

    def to_pil(self, origin='lower'):
        data = self.data
        if cupy:
            data = cupy.asnumpy(data)
        arr = np.flipud(data) if origin == 'lower' else data
        return fromarray(arr, 'RGBA')

    def to_bytesio(self, format='png', origin='lower'):
        fp = BytesIO()
        self.to_pil(origin).save(fp, format)
        fp.seek(0)
        return fp

    def _repr_png_(self):
        """Supports rich PNG display in a Jupyter notebook"""
        return self.to_pil()._repr_png_()

    def _repr_html_(self):
        """Supports rich HTML display in a Jupyter notebook"""
        # imported here to avoid depending on these packages unless actually used
        from io import BytesIO
        from base64 import b64encode

        b = BytesIO()
        self.to_pil().save(b, format='png')

        h = """<img style="margin: auto; border:""" + str(self.border) + """px solid" """ + \
            """src='data:image/png;base64,{0}'/>""".\
                format(b64encode(b.getvalue()).decode('utf-8'))
        return h



class Images(object):
    """
    A list of HTML-representable objects to display in a table.
    Primarily intended for Image objects, but could be anything
    that has _repr_html_.
    """

    def __init__(self, *images):
        """Makes an HTML table from a list of HTML-representable arguments."""
        for i in images:
            assert hasattr(i,"_repr_html_")
        self.images = images
        self.num_cols = None

    def cols(self,n):
        """
        Set the number of columns to use in the HTML table.
        Returns self for convenience.
        """
        self.num_cols=n
        return self

    def _repr_html_(self):
        """Supports rich display in a Jupyter notebook, using an HTML table"""
        htmls = []
        col=0
        tr="""<tr style="background-color:white">"""
        for i in self.images:
            label=i.name if hasattr(i,"name") and i.name is not None else ""

            htmls.append("""<td style="text-align: center"><b>""" + label +
                         """</b><br><br>{0}</td>""".format(i._repr_html_()))
            col+=1
            if self.num_cols is not None and col>=self.num_cols:
                col=0
                htmls.append("</tr>"+tr)

        return """<table style="width:100%; text-align: center"><tbody>"""+ tr +\
               "".join(htmls) + """</tr></tbody></table>"""



def stack(*imgs, **kwargs):
    """Combine images together, overlaying later images onto earlier ones.

    Parameters
    ----------
    imgs : iterable of Image
        The images to combine.
    how : str, optional
        The compositing operator to combine pixels. Default is `'over'`.
    """
    from datashader.composite import composite_op_lookup

    if not imgs:
        raise ValueError("No images passed in")
    shapes = []
    for i in imgs:
        if not isinstance(i, Image):
            raise TypeError("Expected `Image`, got: `{0}`".format(type(i)))
        elif not shapes:
            shapes.append(i.shape)
        elif shapes and i.shape not in shapes:
            raise ValueError("The stacked images must have the same shape.")

    name = kwargs.get('name', None)
    op = composite_op_lookup[kwargs.get('how', 'over')]
    if len(imgs) == 1:
        return imgs[0]
    imgs = xr.align(*imgs, copy=False, join='outer')
    with np.errstate(divide='ignore', invalid='ignore'):
        out = tz.reduce(tz.flip(op), [i.data for i in imgs])
    return Image(out, coords=imgs[0].coords, dims=imgs[0].dims, name=name)


def eq_hist(data, mask=None, nbins=256*256):
    """Return a numpy array after histogram equalization.

    For use in `shade`.

    Parameters
    ----------
    data : ndarray
    mask : ndarray, optional
       Boolean array of missing points. Where True, the output will be `NaN`.
    nbins : int, optional
        Maximum number of bins to use. If data is of type boolean or integer
        this will determine when to switch from exact unique value counts to
        a binned histogram.

    Notes
    -----
    This function is adapted from the implementation in scikit-image [1]_.

    References
    ----------
    .. [1] http://scikit-image.org/docs/stable/api/skimage.exposure.html#equalize-hist
    """
    if cupy and isinstance(data, cupy.ndarray):
        from._cuda_utils import interp
        array_module = cupy
    elif not isinstance(data, np.ndarray):
        raise TypeError("data must be an ndarray")
    else:
        interp = np.interp
        array_module = np

    if mask is not None and array_module.all(mask):
        # Issue #1166, return early with array of all nans if all of data is masked out.
        return array_module.full_like(data, np.nan), 0

    data2 = data if mask is None else data[~mask]

    # Run more accurate value counting if data is of boolean or integer type
    # and unique value array is smaller than nbins.
    if data2.dtype == bool or (array_module.issubdtype(data2.dtype, array_module.integer) and
                               data2.ptp() < nbins):
        values, counts = array_module.unique(data2, return_counts=True)
        vmin, vmax = values[0].item(), values[-1].item()  # Convert from arrays to scalars.
        interval = vmax-vmin
        bin_centers = array_module.arange(vmin, vmax+1)
        hist = array_module.zeros(interval+1, dtype='uint64')
        hist[values-vmin] = counts
        discrete_levels = len(values)
    else:
        hist, bin_edges = array_module.histogram(data2, bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        keep_mask = (hist > 0)
        discrete_levels = array_module.count_nonzero(keep_mask)
        if discrete_levels != len(hist):
            # Remove empty histogram bins.
            hist = hist[keep_mask]
            bin_centers = bin_centers[keep_mask]
    cdf = hist.cumsum()
    cdf = cdf / float(cdf[-1])
    out = interp(data, bin_centers, cdf).reshape(data.shape)
    return out if mask is None else array_module.where(mask, array_module.nan, out), discrete_levels




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


def _rescale_discrete_levels(discrete_levels, span):
    if discrete_levels is None:
        raise ValueError("interpolator did not return a valid discrete_levels")

    # Straight line y = mx + c through (2, 1.5) and (100, 1) where
    # x is number of discrete_levels and y is lower span limit.
    m = -0.5/98.0  # (y[1] - y[0]) / (x[1] - x[0])
    c = 1.5 - 2*m  # y[0] - m*x[0]
    multiple = m*discrete_levels + c

    if multiple > 1:
        lower_span = max(span[1] - multiple*(span[1] - span[0]), 0)
        span = (lower_span, 1)

    return span


def _interpolate(agg, cmap, how, alpha, span, min_alpha, name, rescale_discrete_levels):
    if cupy and isinstance(agg.data, cupy.ndarray):
        from ._cuda_utils import masked_clip_2d, interp
    else:
        from ._cpu_utils import masked_clip_2d
        interp = np.interp

    if agg.ndim != 2:
        raise ValueError("agg must be 2D")
    interpolater = _normalize_interpolate_how(how)

    data = agg.data
    if isinstance(data, da.Array):
        data = data.compute()
    else:
        data = data.copy()

    # Compute mask
    if np.issubdtype(data.dtype, np.bool_):
        mask = ~data
        data = data.astype(np.int8)
    else:
        if data.dtype.kind == 'u':
            mask = data == 0
        else:
            mask = np.isnan(data)

    # Handle case where everything is masked out
    if mask.all():
        return Image(np.zeros(shape=agg.data.shape,
                              dtype=np.uint32), coords=agg.coords,
                     dims=agg.dims, attrs=agg.attrs, name=name)

    # Handle offset / clip
    if span is None:
        offset = np.nanmin(data[~mask])
    else:
        offset = np.array(span, dtype=data.dtype)[0]
        masked_clip_2d(data, mask, *span)

    # If log/cbrt, could case to float64 right away
    # If linear, can keep current type
    data -= offset

    with np.errstate(invalid="ignore", divide="ignore"):
        # Transform data (log, eq_hist, etc.)
        data = interpolater(data, mask)
        discrete_levels = None
        if isinstance(data, (list, tuple)):
            data, discrete_levels = data

        # Transform span
        if span is None:
            masked_data = np.where(~mask, data, np.nan)
            span = np.nanmin(masked_data), np.nanmax(masked_data)

            if rescale_discrete_levels and discrete_levels is not None:  # Only valid for how='eq_hist'
                span = _rescale_discrete_levels(discrete_levels, span)
        else:
            if how == 'eq_hist':
                # For eq_hist to work with span, we'd need to compute the histogram
                # only on the specified span's range.
                raise ValueError("span is not (yet) valid to use with eq_hist")

            span = interpolater([0, span[1] - span[0]], 0)

    if isinstance(cmap, Iterator):
        cmap = list(cmap)
    if isinstance(cmap, tuple) and isinstance(cmap[0], str):
        cmap = list(cmap)  # Tuple of hex values or color names, as produced by Bokeh
    if isinstance(cmap, list):
        rspan, gspan, bspan = np.array(list(zip(*map(rgb, cmap))))
        span = np.linspace(span[0], span[1], len(cmap))
        r = np.nan_to_num(interp(data, span, rspan, left=255), copy=False).astype(np.uint8)
        g = np.nan_to_num(interp(data, span, gspan, left=255), copy=False).astype(np.uint8)
        b = np.nan_to_num(interp(data, span, bspan, left=255), copy=False).astype(np.uint8)
        a = np.where(np.isnan(data), 0, alpha).astype(np.uint8)
        rgba = np.dstack([r, g, b, a])
    elif isinstance(cmap, str) or isinstance(cmap, tuple):
        color = rgb(cmap)
        aspan = np.arange(min_alpha, alpha+1)
        span = np.linspace(span[0], span[1], len(aspan))
        r = np.full(data.shape, color[0], dtype=np.uint8)
        g = np.full(data.shape, color[1], dtype=np.uint8)
        b = np.full(data.shape, color[2], dtype=np.uint8)
        a = np.nan_to_num(interp(data, span, aspan, left=0, right=255), copy=False).astype(np.uint8)
        rgba = np.dstack([r, g, b, a])
    elif callable(cmap):
        # Assume callable is matplotlib colormap
        scaled_data = (data - span[0])/(span[1] - span[0])
        if cupy and isinstance(scaled_data, cupy.ndarray):
            # Convert cupy array to numpy before passing to matplotlib colormap
            scaled_data = cupy.asnumpy(scaled_data)

        rgba = cmap(scaled_data, bytes=True)
        rgba[:, :, 3] = np.where(np.isnan(scaled_data), 0, alpha).astype(np.uint8)
    else:
        raise TypeError("Expected `cmap` of `matplotlib.colors.Colormap`, "
                        "`list`, `str`, or `tuple`; got: '{0}'".format(type(cmap)))

    img = rgba.view(np.uint32).reshape(data.shape)

    if cupy and isinstance(img, cupy.ndarray):
        # Convert cupy array to numpy for final image
        img = cupy.asnumpy(img)

    return Image(img, coords=agg.coords, dims=agg.dims, name=name)


def _colorize(agg, color_key, how, alpha, span, min_alpha, name, color_baseline, rescale_discrete_levels):
    if cupy and isinstance(agg.data, cupy.ndarray):
        array = cupy.array
    else:
        array = np.array

    if not agg.ndim == 3:
        raise ValueError("agg must be 3D")

    cats = agg.indexes[agg.dims[-1]]
    if not len(cats): # No categories and therefore no data; return an empty image
        return Image(np.zeros(agg.shape[0:2], dtype=np.uint32), dims=agg.dims[:-1],
                     coords=dict([
                         (agg.dims[1], agg.coords[agg.dims[1]]),
                         (agg.dims[0], agg.coords[agg.dims[0]]) ]), name=name)

    if color_key is None:
        raise ValueError("Color key must be provided, with at least as many " +
                         "colors as there are categorical fields")
    if not isinstance(color_key, dict):
        color_key = dict(zip(cats, color_key))
    if len(color_key) < len(cats):
        raise ValueError("Insufficient colors provided ({}) for the categorical fields available ({})"
                         .format(len(color_key), len(cats)))

    colors = [rgb(color_key[c]) for c in cats]
    rs, gs, bs = map(array, zip(*colors))

    # Reorient array (transposing the category dimension first)
    agg_t = agg.transpose(*((agg.dims[-1],)+agg.dims[:2]))
    data = agg_t.data.transpose([1, 2, 0])
    if isinstance(data, da.Array):
        data = data.compute()
    color_data = data.copy()

    # subtract color_baseline if needed
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
        baseline = np.nanmin(color_data) if color_baseline is None else color_baseline
    with np.errstate(invalid='ignore'):
        if baseline > 0:
            color_data -= baseline
        elif baseline < 0:
            color_data += -baseline
        if color_data.dtype.kind != 'u' and color_baseline is not None:
            color_data[color_data<0]=0

    color_total = nansum_missing(color_data, axis=2)
    # dot does not handle nans, so replace with zeros
    color_data[np.isnan(data)] = 0

    # zero-count pixels will be 0/0, but it's safe to ignore that when dividing
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (color_data.dot(rs)/color_total).astype(np.uint8)
        g = (color_data.dot(gs)/color_total).astype(np.uint8)
        b = (color_data.dot(bs)/color_total).astype(np.uint8)

    # special case -- to give an appropriate color when min_alpha != 0 and data=0,
    # take avg color of all non-nan categories
    color_mask = ~np.isnan(data)
    cmask_sum = np.sum(color_mask, axis=2)

    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = (color_mask.dot(rs)/cmask_sum).astype(np.uint8)
        g2 = (color_mask.dot(gs)/cmask_sum).astype(np.uint8)
        b2 = (color_mask.dot(bs)/cmask_sum).astype(np.uint8)

    missing_colors = np.sum(color_data, axis=2) == 0
    r = np.where(missing_colors, r2, r)
    g = np.where(missing_colors, g2, g)
    b = np.where(missing_colors, b2, b)

    total = nansum_missing(data, axis=2)
    mask = np.isnan(total)
    a = _interpolate_alpha(data, total, mask, how, alpha, span, min_alpha, rescale_discrete_levels)

    values = np.dstack([r, g, b, a]).view(np.uint32).reshape(a.shape)
    if cupy and isinstance(values, cupy.ndarray):
        # Convert cupy array to numpy for final image
        values = cupy.asnumpy(values)

    return Image(values,
                 dims=agg.dims[:-1],
                 coords=dict([
                     (agg.dims[1], agg.coords[agg.dims[1]]),
                     (agg.dims[0], agg.coords[agg.dims[0]]),
                 ]),
                 name=name)


def _interpolate_alpha(data, total, mask, how, alpha, span, min_alpha, rescale_discrete_levels):

    if cupy and isinstance(data, cupy.ndarray):
        from ._cuda_utils import interp, masked_clip_2d
        array_module = cupy
    else:
        from ._cpu_utils import masked_clip_2d
        interp = np.interp
        array_module = np

    # if span is provided, use it, otherwise produce a span based off the
    # min/max of the data
    if span is None:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            offset = np.nanmin(total)
        if total.dtype.kind == 'u' and offset == 0:
            mask = mask | (total == 0)
            # If at least one element is not masked, use the minimum as the offset
            # otherwise the offset remains at zero
            if not np.all(mask):
                offset = total[total > 0].min()
            total = np.where(~mask, total, np.nan)

        a_scaled = _normalize_interpolate_how(how)(total - offset, mask)
        discrete_levels = None
        if isinstance(a_scaled, (list, tuple)):
            a_scaled, discrete_levels = a_scaled

        # All-NaN objects (e.g. chunks of arrays with no data) are valid in Datashader
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            norm_span = [np.nanmin(a_scaled).item(), np.nanmax(a_scaled).item()]

        if rescale_discrete_levels and discrete_levels is not None:  # Only valid for how='eq_hist'
            norm_span = _rescale_discrete_levels(discrete_levels, norm_span)

    else:
        if how == 'eq_hist':
            # For eq_hist to work with span, we'll need to compute the histogram
            # only on the specified span's range.
            raise ValueError("span is not (yet) valid to use with eq_hist")
        # even in fixed-span mode cells with 0 should remain fully transparent
        # i.e. a 0 will be fully transparent, but any non-zero number will
        # be clipped to the span range and have min-alpha applied
        offset = np.array(span, dtype=data.dtype)[0]
        if total.dtype.kind == 'u' and np.nanmin(total) == 0:
            mask = mask | (total <= 0)
            total = np.where(~mask, total, np.nan)
        masked_clip_2d(total, mask, *span)

        a_scaled = _normalize_interpolate_how(how)(total - offset, mask)
        if isinstance(a_scaled, (list, tuple)):
            a_scaled = a_scaled[0]  # Ignore discrete_levels

        norm_span = _normalize_interpolate_how(how)([0, span[1] - span[0]], 0)
        if isinstance(norm_span, (list, tuple)):
            norm_span = norm_span[0]  # Ignore discrete_levels

    # Issue 1178. Convert norm_span from 2-tuple to numpy/cupy array.
    # array_module.hstack() tolerates tuple of one float and one cupy array,
    # whereas array_module.array() does not.
    norm_span = array_module.hstack(norm_span)

    # Interpolate the alpha values
    a_float = interp(a_scaled, norm_span, array_module.array([min_alpha, alpha]), left=0, right=255)
    a = np.nan_to_num(a_float, copy=False).astype(np.uint8)
    return a


def _apply_discrete_colorkey(agg, color_key, alpha, name, color_baseline):
    # use the same approach as 3D case

    if cupy and isinstance(agg.data, cupy.ndarray):
        module = cupy
        array = cupy.array
    else:
        module = np
        array = np.array

    if not agg.ndim == 2:
        raise ValueError("agg must be 2D")

    # validate color_key
    if (color_key is None) or (not isinstance(color_key, dict)):
        raise ValueError("Color key must be provided as a dictionary")

    agg_data = agg.data
    if isinstance(agg_data, da.Array):
        agg_data = agg_data.compute()

    cats = color_key.keys()
    colors = [rgb(color_key[c]) for c in cats]
    rs, gs, bs = map(array, zip(*colors))

    data = module.empty_like(agg_data) * module.nan

    r = module.zeros_like(data, dtype=module.uint8)
    g = module.zeros_like(data, dtype=module.uint8)
    b = module.zeros_like(data, dtype=module.uint8)

    r2 = module.zeros_like(data, dtype=module.uint8)
    g2 = module.zeros_like(data, dtype=module.uint8)
    b2 = module.zeros_like(data, dtype=module.uint8)

    for i, c in enumerate(cats):
        value_mask = agg_data == c
        data[value_mask] = 1
        r2[value_mask] = rs[i]
        g2[value_mask] = gs[i]
        b2[value_mask] = bs[i]

    color_data = data.copy()

    # subtract color_baseline if needed
    baseline = module.nanmin(color_data) if color_baseline is None else color_baseline
    with np.errstate(invalid='ignore'):
        if baseline > 0:
            color_data -= baseline
        elif baseline < 0:
            color_data += -baseline
        if color_data.dtype.kind != 'u' and color_baseline is not None:
            color_data[color_data < 0] = 0

    color_data[module.isnan(data)] = 0
    if not color_data.any():
        r[:] = r2
        g[:] = g2
        b[:] = b2

    missing_colors = color_data == 0
    r = module.where(missing_colors, r2, r)
    g = module.where(missing_colors, g2, g)
    b = module.where(missing_colors, b2, b)

    # alpha channel
    a = np.where(np.isnan(data), 0, alpha).astype(np.uint8)

    values = module.dstack([r, g, b, a]).view(module.uint32).reshape(a.shape)

    if cupy and isinstance(agg.data, cupy.ndarray):
        # Convert cupy array to numpy for final image
        values = cupy.asnumpy(values)

    return Image(values,
                 dims=agg.dims,
                 coords=agg.coords,
                 name=name
                 )


def shade(agg, cmap=["lightblue", "darkblue"], color_key=Sets1to3,
          how='eq_hist', alpha=255, min_alpha=40, span=None, name=None,
          color_baseline=None, rescale_discrete_levels=False):
    """Convert a DataArray to an image by choosing an RGBA pixel color for each value.

    Requires a DataArray with a single data dimension, here called the
    "value", indexed using either 2D or 3D coordinates.

    For a DataArray with 2D coordinates, the RGB channels are computed
    from the values by interpolated lookup into the given colormap
    ``cmap``.  The A channel is then set to the given fixed ``alpha``
    value for all non-zero values, and to zero for all zero values.
    A dictionary ``color_key`` that specifies categories (values in ``agg``)
    and corresponding colors can be provided to support discrete coloring
    2D aggregates, i.e aggregates with a single category per pixel,
    with no mixing. The A channel is set the given ``alpha`` value for all
    pixels in the categories specified in ``color_key``, and to zero otherwise.

    DataArrays with 3D coordinates are expected to contain values
    distributed over different categories that are indexed by the
    additional coordinate. Such an array would reduce to the
    2D-coordinate case if collapsed across the categories (e.g. if one
    did ``aggc.sum(dim='cat')`` for a categorical dimension ``cat``).
    The RGB channels for the uncollapsed, 3D case are mixed from separate
    values over all categories. They are computed by averaging the colors
    in the provided ``color_key`` (with one color per category),
    weighted by the array's value for that category.
    The A channel is then computed from the array's total value
    collapsed across all categories at that location, ranging from the
    specified ``min_alpha`` to the maximum alpha value (255).

    Parameters
    ----------
    agg : DataArray
    cmap : list of colors or matplotlib.colors.Colormap, optional
        The colormap to use for 2D agg arrays. Can be either a list of
        colors (specified either by name, RGBA hexcode, or as a tuple
        of ``(red, green, blue)`` values.), or a matplotlib colormap
        object.  Default is ``["lightblue", "darkblue"]``.
    color_key : dict or iterable
        The colors to use for a categorical agg array. In 3D case, it can be
        either a ``dict`` mapping from field name to colors, or an
        iterable of colors in the same order as the record fields,
        and including at least that many distinct colors. In 2D case,
        ``color_key`` must be a ``dict`` where all keys are categories,
        and values are corresponding colors. Number of categories does not
        necessarily equal to the number of unique values in the agg DataArray.
    how : str or callable, optional
        The interpolation method to use, for the ``cmap`` of a 2D
        DataArray or the alpha channel of a 3D DataArray. Valid
        strings are 'eq_hist' [default], 'cbrt' (cube root), 'log'
        (logarithmic), and 'linear'. Callables take 2 arguments - a
        2-dimensional array of magnitudes at each pixel, and a boolean
        mask array indicating missingness. They should return a numeric
        array of the same shape, with ``NaN`` values where the mask was
        True.
    alpha : int, optional
        Value between 0 - 255 representing the alpha value to use for
        colormapped pixels that contain data (i.e. non-NaN values).
        Also used as the maximum alpha value when alpha is indicating
        data value, such as for single colors or categorical plots.
        Regardless of this value, ``NaN`` values are set to be fully
        transparent when doing colormapping.
    min_alpha : float, optional
        The minimum alpha value to use for non-empty pixels when
        alpha is indicating data value, in [0, 255].  Use a higher value
        to avoid undersaturation, i.e. poorly visible low-value datapoints,
        at the expense of the overall dynamic range. Note that ``min_alpha``
        will not take any effect when doing discrete categorical coloring
        for 2D case as the aggregate can have only a single value to denote
        the category.
    span : list of min-max range, optional
        Min and max data values to use for 2D colormapping,
        and 3D alpha interpolation, when wishing to override autoranging.
    name : string name, optional
        Optional string name to give to the Image object to return,
        to label results for display.
    color_baseline : float or None
        Baseline for calculating how categorical data mixes to
        determine the color of a pixel. The color for each category is
        weighted by how far that category's value is above this
        baseline value, out of the total sum across all categories'
        values. A value of zero is appropriate for counts and for
        other physical quantities for which zero is a meaningful
        reference; each category then contributes to the final color
        in proportion to how much each category contributes to the
        final sum.  However, if values can be negative or if they are
        on an interval scale where values e.g. twice as far from zero
        are not twice as high (such as temperature in Farenheit), then
        you will need to provide a suitable baseline value for use in
        calculating color mixing.  A value of None (the default) means
        to take the minimum across the entire aggregate array, which
        is safe but may not weight the colors as you expect; any
        categories with values near this baseline will contribute
        almost nothing to the final color. As a special case, if the
        only data present in a pixel is at the baseline level, the
        color will be an evenly weighted average of all such
        categories with data (to avoid the color being undefined in
        this case).
    rescale_discrete_levels : boolean, optional
        If ``how='eq_hist`` and there are only a few discrete values,
        then ``rescale_discrete_levels=True`` decreases the lower
        limit of the autoranged span so that the values are rendering
        towards the (more visible) top of the ``cmap`` range, thus
        avoiding washout of the lower values.  Has no effect if
        ``how!=`eq_hist``. Default is False.
    """
    if not isinstance(agg, xr.DataArray):
        raise TypeError("agg must be instance of DataArray")
    name = agg.name if name is None else name

    if not ((0 <= min_alpha <= 255) and (0 <= alpha <= 255)):
        raise ValueError("min_alpha ({}) and alpha ({}) must be between 0 and 255".format(min_alpha,alpha))

    if rescale_discrete_levels and how != 'eq_hist':
        rescale_discrete_levels = False

    if agg.ndim == 2:
        if color_key is not None and isinstance(color_key, dict):
            return _apply_discrete_colorkey(
                agg, color_key, alpha, name, color_baseline
            )
        else:
            return _interpolate(agg, cmap, how, alpha, span, min_alpha, name, rescale_discrete_levels)
    elif agg.ndim == 3:
        return _colorize(agg, color_key, how, alpha, span, min_alpha, name, color_baseline, rescale_discrete_levels)
    else:
        raise ValueError("agg must use 2D or 3D coordinates")


def set_background(img, color=None, name=None):
    """Return a new image, with the background set to `color`.

    Parameters
    ----------
    img : Image
    color : color name or tuple, optional
        The background color. Can be specified either by name, hexcode, or as a
        tuple of ``(red, green, blue)`` values.
    """
    from datashader.composite import over

    if not isinstance(img, Image):
        raise TypeError("Expected `Image`, got: `{0}`".format(type(img)))
    name = img.name if name is None else name
    if color is None:
        return img
    background = np.uint8(rgb(color) + (255,)).view('uint32')[0]
    data = over(img.data, background)
    return Image(data, coords=img.coords, dims=img.dims, name=name)


def spread(img, px=1, shape='circle', how=None, mask=None, name=None):
    """Spread pixels in an image.

    Spreading expands each pixel a certain number of pixels on all sides
    according to a given shape, merging pixels using a specified compositing
    operator. This can be useful to make sparse plots more visible.

    Parameters
    ----------
    img : Image or other DataArray
    px : int, optional
        Number of pixels to spread on all sides
    shape : str, optional
        The shape to spread by. Options are 'circle' [default] or 'square'.
    how : str, optional
        The name of the compositing operator to use when combining
        pixels. Default of None uses 'over' operator for Image objects
        and 'add' operator otherwise.
    mask : ndarray, shape (M, M), optional
        The mask to spread over. If provided, this mask is used instead of
        generating one based on `px` and `shape`. Must be a square array
        with odd dimensions. Pixels are spread from the center of the mask to
        locations where the mask is True.
    name : string name, optional
        Optional string name to give to the Image object to return,
        to label results for display.
    """
    if not isinstance(img, xr.DataArray):
        raise TypeError("Expected `xr.DataArray`, got: `{0}`".format(type(img)))
    is_image = isinstance(img, Image)
    name = img.name if name is None else name
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
    if how is None:
        how = 'over' if is_image else 'add'

    w = mask.shape[0]
    extra = w // 2
    M, N = img.shape[:2]
    padded_shape = (M + 2*extra, N + 2*extra)
    float_type = img.dtype in [np.float32, np.float64]
    fill_value = np.nan if float_type else 0
    if cupy and isinstance(img.data, cupy.ndarray):
        # Convert img.data to numpy array before passing to nb.jit kernels
        img.data = cupy.asnumpy(img.data)

    if is_image:
        kernel = _build_spread_kernel(how, is_image)
    elif float_type:
        kernel = _build_float_kernel(how, w)
    else:
        kernel = _build_int_kernel(how, w, img.dtype == np.uint32)

    def apply_kernel(layer):
        buf = np.full(padded_shape, fill_value, dtype=layer.dtype)
        kernel(layer.data, mask, buf)
        return buf[extra:-extra, extra:-extra].copy()

    if len(img.shape)==2:
        out = apply_kernel(img)
    else:
        out = np.dstack([apply_kernel(img[:,:,category])
                        for category in range(img.shape[2])])

    return img.__class__(out, dims=img.dims, coords=img.coords, name=name)


@tz.memoize
def _build_int_kernel(how, mask_size, ignore_zeros):
    """Build a spreading kernel for a given composite operator"""
    from datashader.composite import composite_op_lookup, validate_operator

    validate_operator(how, is_image=False)
    op = composite_op_lookup[how + "_arr"]
    @ngjit
    def stencilled(arr, mask, out):
        M, N = arr.shape
        for y in range(M):
            for x in range(N):
                el = arr[y, x]
                for i in range(mask_size):
                    for j in range(mask_size):
                        if mask[i, j]:
                            if ignore_zeros and el==0:
                                result = out[i + y, j + x]
                            elif ignore_zeros and out[i + y, j + x]==0:
                                result = el
                            else:
                                result = op(el, out[i + y, j + x])
                            out[i + y, j + x] = result
    return stencilled


@tz.memoize
def _build_float_kernel(how, mask_size):
    """Build a spreading kernel for a given composite operator"""
    from datashader.composite import composite_op_lookup, validate_operator

    validate_operator(how, is_image=False)
    op = composite_op_lookup[how + "_arr"]
    @ngjit
    def stencilled(arr, mask, out):
        M, N = arr.shape
        for y in range(M):
            for x in range(N):
                el = arr[y, x]
                for i in range(mask_size):
                    for j in range(mask_size):
                        if mask[i, j]:
                            if np.isnan(el):
                                result = out[i + y, j + x]
                            elif np.isnan(out[i + y, j + x]):
                                result = el
                            else:
                                result = op(el, out[i + y, j + x])
                            out[i + y, j + x] = result
    return stencilled


@tz.memoize
def _build_spread_kernel(how, is_image):
    """Build a spreading kernel for a given composite operator"""
    from datashader.composite import composite_op_lookup, validate_operator

    validate_operator(how, is_image=True)
    op = composite_op_lookup[how + ("" if is_image else "_arr")]

    @ngjit
    def kernel(arr, mask, out):
        M, N = arr.shape
        w = mask.shape[0]
        for y in range(M):
            for x in range(N):
                el = arr[y, x]
                # Skip if data is transparent
                process_image = is_image and ((int(el) >> 24) & 255) # Transparent pixel
                process_array = (not is_image) and (not np.isnan(el))
                if process_image or process_array:
                    for i in range(w):
                        for j in range(w):
                            # Skip if mask is False at this value
                            if mask[i, j]:
                                if el==0:
                                    result = out[i + y, j + x]
                                if out[i + y, j + x]==0:
                                    result = el
                                else:
                                    result = op(el, out[i + y, j + x])
                                out[i + y, j + x] = result
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


def dynspread(img, threshold=0.5, max_px=3, shape='circle', how=None, name=None):
    """Spread pixels in an image dynamically based on the image density.

    Spreading expands each pixel a certain number of pixels on all sides
    according to a given shape, merging pixels using a specified compositing
    operator. This can be useful to make sparse plots more visible. Dynamic
    spreading determines how many pixels to spread based on a density
    heuristic.  Spreading starts at 1 pixel, and stops when the fraction
    of adjacent non-empty pixels reaches the specified threshold, or
    the max_px is reached, whichever comes first.

    Parameters
    ----------
    img : Image
    threshold : float, optional
        A tuning parameter in [0, 1], with higher values giving more
        spreading.
    max_px : int, optional
        Maximum number of pixels to spread on all sides.
    shape : str, optional
        The shape to spread by. Options are 'circle' [default] or 'square'.
    how : str, optional
        The name of the compositing operator to use when combining
        pixels. Default of None uses 'over' operator for Image objects
        and 'add' operator otherwise.
    """
    is_image = isinstance(img, Image)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if not isinstance(max_px, int) or max_px < 0:
        raise ValueError("max_px must be >= 0")
    # Simple linear search. Not super efficient, but max_px is usually small.
    float_type = img.dtype in [np.float32, np.float64]
    if cupy and isinstance(img.data, cupy.ndarray):
        # Convert img.data to numpy array before passing to nb.jit kernels
        img.data = cupy.asnumpy(img.data)

    px_=0
    for px in range(1, max_px + 1):
        px_=px
        if is_image:
            density = _rgb_density(img.data, px*2)
        elif len(img.shape) == 2:
            density = _array_density(img.data, float_type, px*2)
        else:
            masked = np.logical_not(np.isnan(img)) if float_type else (img != 0)
            flat_mask = np.sum(masked, axis=2, dtype='uint32')
            density = _array_density(flat_mask.data, False, px*2)
        if density > threshold:
            px_=px_-1
            break

    if px_>=1:
        return spread(img, px_, shape=shape, how=how, name=name)
    else:
        return img


@nb.jit(nopython=True, nogil=True, cache=True)
def _array_density(arr, float_type, px=1):
    """Compute a density heuristic of an array.

    The density is a number in [0, 1], and indicates the normalized mean number
    of non-empty pixels that have neighbors in the given px radius.
    """
    M, N = arr.shape
    cnt = has_neighbors = 0
    for y in range(0, M):
        for x in range(0, N):
            el = arr[y, x]
            if (float_type and not np.isnan(el)) or (not float_type and el!=0):
                cnt += 1
                neighbors = 0
                for i in     range(max(0, y - px), min(y + px + 1, M)):
                    for j in range(max(0, x - px), min(x + px + 1, N)):
                        if ((float_type and not np.isnan(arr[i, j])) or
                            (not float_type and arr[i, j] != 0)):
                            neighbors += 1
                if neighbors>1: # (excludes self)
                    has_neighbors += 1
    return has_neighbors/cnt if cnt else np.inf


@nb.jit(nopython=True, nogil=True, cache=True)
def _rgb_density(arr, px=1):
    """Compute a density heuristic of an image.

    The density is a number in [0, 1], and indicates the normalized mean number
    of non-empty pixels that have neighbors in the given px radius.
    """
    M, N = arr.shape
    cnt = has_neighbors = 0
    for y in range(0, M):
        for x in range(0, N):
            if (arr[y, x] >> 24) & 255:
                cnt += 1
                neighbors = 0
                for i in     range(max(0, y - px), min(y + px + 1, M)):
                    for j in range(max(0, x - px), min(x + px + 1, N)):
                        if (arr[i, j] >> 24) & 255:
                            neighbors += 1
                if neighbors>1: # (excludes self)
                    has_neighbors += 1
    return has_neighbors/cnt if cnt else np.inf
