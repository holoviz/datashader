from collections import OrderedDict
import warnings

from matplotlib.image import _ImageBase
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform
import matplotlib as mpl
import numpy as np

from . import reductions
from . import transfer_functions as tf
from .colors import Sets1to3
from .core import bypixel, Canvas

__all__ = ["ScalarDSArtist", "CategoricalDSArtist", "alpha_colormap", "dsshow"]


def uint32_to_uint8(img):
    """Cast a uint32 raster to a 4-channel uint8 RGBA array."""
    return img.view(dtype=np.uint8).reshape(img.shape + (4,))


def uint8_to_uint32(img):
    """Cast a 4-channel uint8 RGBA array to uint32 raster"""
    return img.view(dtype=np.uint32).reshape(img.shape[:-1])


def to_ds_image(binned, rgba):
    if binned.ndim == 2:
        return tf.Image(uint8_to_uint32(rgba), coords=binned.coords, dims=binned.dims)
    elif binned.ndim == 3:
        return tf.Image(
            uint8_to_uint32(rgba),
            dims=binned.dims[:-1],
            coords=OrderedDict(
                [
                    (binned.dims[1], binned.coords[binned.dims[1]]),
                    (binned.dims[0], binned.coords[binned.dims[0]]),
                ]
            ),
        )
    else:
        raise ValueError("Aggregate must be 2D or 3D.")


def compute_mask(binned):
    # Use datashader's rules for masking aggregates
    # mask == True --> invalid
    if np.issubdtype(binned.dtype, np.bool_):
        mask = ~binned
    elif binned.dtype.kind == "u":
        mask = binned == 0
    else:
        mask = np.isnan(binned)
    return mask


def alpha_colormap(color, min_alpha=40, max_alpha=255, N=256):
    """
    Generate a transparency-based monochromatic colormap.

    Parameters
    ----------
    color : str or tuple
        Color name, hex code or RGB tuple.
    min_alpha, max_alpha: int
        Values between 0 - 255 representing the range of alpha values to use for
        colormapped pixels that contain data.

    Returns
    -------
    :class:`matplotlib.colors.LinearSegmentedColormap`

    """
    for a in (min_alpha, max_alpha):
        if a < 0 or a > 255:
            raise ValueError("Alpha values must be integers between 0 and 255")
    r, g, b = mpl.colors.to_rgb(color)
    return mpl.colors.LinearSegmentedColormap(
        "_datashader_alpha",
        {
            "red": [(0.0, r, r), (1.0, r, r)],
            "green": [(0.0, g, g), (1.0, g, g)],
            "blue": [(0.0, b, b), (1.0, b, b)],
            "alpha": [
                (0.0, min_alpha / 255, min_alpha / 255),
                (1.0, max_alpha / 255, max_alpha / 255),
            ],
        },
        N=N,
    )


class EqHistNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False, nbins=256 ** 2, ncolors=256):
        super(EqHistNormalize, self).__init__(vmin, vmax, clip)
        self._nbins = nbins
        self._bin_edges = None
        self._ncolors = ncolors
        self._color_bins = np.linspace(0, 1, ncolors)

    def _binning(self, data, n=256):
        if np.ma.is_masked(data):
            data = data[~data.mask]

        low = data.min() if self.vmin is None else self.vmin
        high = data.max() if self.vmax is None else self.vmax
        nbins = self._nbins
        eq_bin_edges = np.linspace(low, high, nbins + 1)
        hist, _ = np.histogram(data, eq_bin_edges)

        eq_bin_centers = np.convolve(eq_bin_edges, [0.5, 0.5], mode="valid")
        cdf = np.cumsum(hist)
        cdf_max = cdf[-1]
        norm_cdf = cdf / cdf_max

        # Iteratively find as many finite bins as there are colors
        finite_bins = n - 1
        binning = []
        iterations = 0
        guess = n * 2
        while (finite_bins != n) and (iterations < 4) and (finite_bins != 0):
            ratio = guess / finite_bins
            if ratio > 1000:
                # Abort if distribution is extremely skewed
                break
            guess = np.round(max(n * ratio, n))

            # Interpolate
            palette_edges = np.arange(0, guess)
            palette_cdf = norm_cdf * (guess - 1)
            binning = np.interp(palette_edges, palette_cdf, eq_bin_centers)

            # Evaluate binning
            uniq_bins = np.unique(binning)
            finite_bins = len(uniq_bins) - 1
            iterations += 1
        if finite_bins == 0:
            binning = [low] + [high] * (n - 1)
        else:
            binning = binning[-n:]
            if finite_bins != n:
                warnings.warn(
                    "EqHistColorMapper warning: Histogram equalization did not converge."
                )
        return binning

    def __call__(self, data, clip=None):
        mask = np.ma.getmask(data)
        result = self.process_value(data)[0]
        # Preserve the mask after normalization if there is one
        return np.ma.masked_array(result, mask)

    def process_value(self, data):
        isscalar = np.isscalar(data)
        data = np.array([data]) if isscalar else data
        interped = np.interp(data, self._bin_edges, self._color_bins)
        return interped, isscalar

    def inverse(self, value):
        if self._bin_edges is None:
            raise ValueError("Not invertible until eq_hist has been computed")
        return np.interp([value], self._color_bins, self._bin_edges)[0]

    def autoscale(self, A):
        super(EqHistNormalize, self).autoscale(A)
        self._bin_edges = self._binning(A, self._ncolors)

    def autoscale_None(self, A):
        super(EqHistNormalize, self).autoscale_None(A)
        self._bin_edges = self._binning(A, self._ncolors)

    def scaled(self):
        return super(EqHistNormalize, self).scaled() and self._bin_edges is not None


class DSArtist(_ImageBase):
    def __init__(
        self,
        ax,
        df,
        glyph,
        reduction,
        on_agg,
        on_color,
        width_scale,
        height_scale,
        initial_x_range,
        initial_y_range,
        origin="lower",
        interpolation="none",
        **kwargs
    ):
        super().__init__(ax, origin=origin, interpolation=interpolation, **kwargs)
        self.axes = ax
        self.df = df
        self.glyph = glyph
        self.reduction = reduction
        self.on_agg = on_agg
        self.on_color = on_color
        self.width_scale = width_scale
        self.height_scale = height_scale
        if initial_x_range is None:
            x_col = glyph.x_label
            initial_x_range = (df[x_col].min(), df[x_col].max())
        if initial_y_range is None:
            y_col = glyph.y_label
            initial_y_range = (df[y_col].min(), df[y_col].max())
        ax.set_xlim(initial_x_range)
        ax.set_ylim(initial_y_range)

    def aggregate(self, x_range, y_range):
        """Aggregate data in the given bounding box to the window dimensions."""
        dims = self.axes.patch.get_window_extent().bounds
        plot_width = int(dims[2] + 0.5)
        plot_height = int(dims[3] + 0.5)

        # Aggregate
        canvas = Canvas(
            plot_width=int(plot_width * self.width_scale),
            plot_height=int(plot_height * self.height_scale),
            x_range=x_range,
            y_range=y_range,
        )
        binned = bypixel(self.df, canvas, self.glyph, self.reduction)

        # Post-aggregation callback
        if self.on_agg is not None:
            binned = self.on_agg(binned)

        return binned

    def colorize(self, binned):
        """Convert an aggregate into an RGBA array."""
        raise NotImplementedError

    def make_image(self, renderer, magnification=1.0, unsampled=True):
        """
        Normalize, rescale, and colormap this image's data for rendering using
        *renderer*, with the given *magnification*.

        If *unsampled* is True, the image will not be scaled, but an
        appropriate affine transformation will be returned instead.

        Returns
        -------
        image : (M, N, 4) uint8 array
            The RGBA image, resampled unless *unsampled* is True.
        x, y : float
            The upper left corner where the image should be drawn, in pixel
            space.
        trans : Affine2D
            The affine transformation from image to pixel space.
        """
        x1, x2, y1, y2 = self.get_extent()
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        trans = self.get_transform()
        transformed_bbox = TransformedBbox(bbox, trans)

        # Generate and save the aggregate
        binned = self.aggregate([x1, x2], [y1, y2])
        self.set_ds_data(binned)

        # Normalize and color to make an RGBA array
        rgba = self.colorize(binned)

        # Post-colorization callback
        if self.on_color is not None:
            img = to_ds_image(binned, rgba)
            img = self.on_color(img)
            rgba = uint32_to_uint8(img.data)

        self.set_array(rgba)

        return self._make_image(
            rgba,
            bbox,
            transformed_bbox,
            self.axes.bbox,
            magnification=magnification,
            unsampled=unsampled,
        )

    def set_ds_data(self, binned):
        """
        Set the aggregate data for the bounding box currently displayed.
        Should be a :class:`xarray.DataArray`.
        """
        self._ds_data = binned

    def get_ds_data(self):
        """
        Return the aggregated, pre-shaded :class:`xarray.DataArray` backing the
        bounding box currently displayed.
        """
        return self._ds_data

    def get_extent(self):
        """Return the image extent as tuple (left, right, bottom, top)"""
        (x1, x2), (y1, y2) = self.axes.get_xlim(), self.axes.get_ylim()
        return x1, x2, y1, y2

    def get_cursor_data(self, event):
        """
        Return the aggregated data at the event position or *None* if the
        event is outside the bounds of the current view.
        """
        xmin, xmax, ymin, ymax = self.get_extent()
        if self.origin == "upper":
            ymin, ymax = ymax, ymin

        arr = self.get_ds_data().data
        data_extent = Bbox([[ymin, xmin], [ymax, xmax]])
        array_extent = Bbox([[0, 0], arr.shape[:2]])
        trans = BboxTransform(boxin=data_extent, boxout=array_extent)

        y, x = event.ydata, event.xdata
        i, j = trans.transform_point([y, x]).astype(int)

        # Clip the coordinates at array bounds
        if not (0 <= i < arr.shape[0]) or not (0 <= j < arr.shape[1]):
            return None
        else:
            return arr[i, j]


class ScalarDSArtist(DSArtist):
    def __init__(
        self,
        ax,
        df,
        glyph,
        reduction,
        on_agg=None,
        on_color=None,
        width_scale=1.0,
        height_scale=1.0,
        initial_x_range=None,
        initial_y_range=None,
        norm=None,
        cmap=None,
        alpha=None,
        **kwargs
    ):
        super().__init__(
            ax,
            df,
            glyph,
            reduction,
            on_agg,
            on_color,
            width_scale,
            height_scale,
            initial_x_range,
            initial_y_range,
            **kwargs
        )
        self._vmin = norm.vmin
        self._vmax = norm.vmax
        self.set_norm(norm)
        self.set_cmap(cmap)
        self.set_alpha(alpha)

        # Aggregate the current view
        binned = self.aggregate(self.axes.get_xlim(), self.axes.get_ylim())
        self.set_ds_data(binned)

        # Placeholder until self.make_image
        self.set_array(np.eye(2))

    def colorize(self, binned):
        # Mask missing data in the greyscale array
        mask = compute_mask(binned.data)
        A = np.ma.masked_array(binned.data, mask)

        # Rescale the norm to the current array
        self.set_array(A)
        self.norm.vmin = self._vmin
        self.norm.vmax = self._vmax
        self.autoscale_None()

        # Make the image with matplotlib
        return self.to_rgba(A, bytes=True, norm=True)

    def get_ds_image(self):
        binned = self.get_ds_data()
        rgba = self.to_rgba(self.get_array(), bytes=True)
        return to_ds_image(binned, rgba)

    def get_legend_elements(self):
        return None


class CategoricalDSArtist(DSArtist):
    def __init__(
        self,
        ax,
        df,
        glyph,
        reduction,
        on_agg=None,
        on_color=None,
        width_scale=1.0,
        height_scale=1.0,
        initial_x_range=None,
        initial_y_range=None,
        color_key=None,
        alpha_range=(40, 255),
        color_baseline=None,
        **kwargs
    ):
        super().__init__(
            ax,
            df,
            glyph,
            reduction,
            on_agg,
            on_color,
            width_scale,
            height_scale,
            initial_x_range,
            initial_y_range,
            **kwargs
        )
        self._color_key = color_key
        self._alpha_range = alpha_range
        self._color_baseline = color_baseline

        # Aggregate the current view
        binned = self.aggregate(self.axes.get_xlim(), self.axes.get_ylim())
        self.set_ds_data(binned)

        # Placeholder until self.make_image
        self.set_array(np.eye(2))

    def colorize(self, binned):
        # Make the blended image with datashader
        img = tf.shade(
            binned,
            color_key=self._color_key,
            min_alpha=self._alpha_range[0],
            alpha=self._alpha_range[1],
            color_baseline=self._color_baseline,
        )
        rgba = uint32_to_uint8(img.data)
        return rgba

    def get_ds_image(self):
        binned = self.get_ds_data()
        rgba = self.get_array()
        return to_ds_image(binned, rgba)

    def get_legend_elements(self):
        """
        Return legend elements to display the color code for each category.
        """
        binned = self.get_ds_data()
        name = binned.dims[2]
        categories = binned.coords[name].data
        color_dict = dict(zip(categories, self._color_key))
        return [
            Patch(facecolor=color, edgecolor="none", label=name)
            for name, color in color_dict.items()
        ]


def dsshow(
    df,
    glyph,
    reduction=reductions.count(),
    on_agg=None,
    on_color=None,
    width_scale=1.0,
    height_scale=1.0,
    *,
    norm=None,
    cmap=None,
    alpha=None,
    vmin=None,
    vmax=None,
    color_key=Sets1to3,
    alpha_range=(40, 255),
    color_baseline=None,
    ax=None,
    fignum=None,
    aspect=None,
    **kwargs
):
    """
    Display the output of a datashading pipeline applied to a dataframe.

    The plot will respond to changes in the bounding box being displayed (in
    data coordinates), such as pan/zoom events. Both scalar and categorical
    datashading pipelines are supported.

    Parameters
    ----------
    df : pandas.DataFrame, dask.DataFrame
        Dataframe to apply the datashading pipeline to.
    glyph : Glyph
        The glyph to bin by.
    reduction : Reduction, optional, default: :class:`~.count`
        The reduction to compute per-pixel.
    on_agg : callable, optional
        A callable that takes the computed aggregate as an argument, and
        returns another aggregate. This can be used to do preprocessing before
        the aggregate is converted to an image.
    on_color : callable, optional
        A callable that takes the image output of the shading pipeline, and
        returns another :class:`~.Image` object. See :func:`~.dynspread` and
        :func:`~.spread` for examples.
    height_scale: float, optional
        Factor by which to scale the height of the image in pixels relative to
        the height of the display space in pixels.
    width_scale: float, optional
        Factor by which to scale the width of the image in pixels relative to
        the width of the display space in pixels.
    norm : str or :class:`matplotlib.colors.Normalize`, optional
        For scalar aggregates, a matplotlib norm to normalize the
        aggregate data to [0, 1] before colormapping. The datashader arguments
        'linear', 'log', 'cbrt' and 'eq_hist' are also supported and correspond
        to equivalent matplotlib norms. Default is the linear norm.
    cmap : str or list or :class:`matplotlib.cm.Colormap`, optional
        For scalar aggregates, a matplotlib colormap name or instance.
        Alternatively, an iterable of colors can be passed and will be converted
        to a colormap. For a single-color, transparency-based colormap, see
        :func:`alpha_colormap`.
    alpha : float
        For scalar aggregates, the alpha blending value, between 0
        (transparent) and 1 (opaque).
    vmin, vmax : float, optional
        For scalar aggregates, the data range that the colormap covers.
        If vmin or vmax is None (default), the colormap autoscales to the
        range of data in the area displayed, unless the corresponding value is
        already set in the norm.
    color_key : dict or iterable, optional
        For categorical aggregates, the colors to use for blending categories.
        See `tf.shade`.
    alpha_range : pair of int, optional
        For categorical aggregates, the minimum and maximum alpha values in
        [0, 255] to use to indicate data values of non-empty pixels. The
        default range is (40, 255).
    color_baseline : float, optional
        For categorical aggregates, the baseline for calculating how
        categorical data mixes to determine the color of a pixel. See
        `tf.shade` for more information.

    Other Parameters
    ----------------
    ax : `matplotlib.Axes`, optional
        Axes to draw into. If *None*, create a new figure or use ``fignum`` to
        draw into an existing figure.
    fignum : None or int or False, optional
        If *None* and ``ax`` is *None*, create a new figure window with
        automatic numbering.
        If a nonzero integer and ``ax`` is *None*, draw into the figure with
        the given number (create it if it does not exist).
        If 0, use the current axes (or create one if it does not exist).
    aspect : {'equal', 'auto'} or float, default: ``rcParams["image.aspect"]``
        The aspect ratio of the axes.
    **kwargs
        All other kwargs are passed to the artist.

    Returns
    -------
    :class:`ScalarDSArtist` or :class:`CategoricalDSArtist`

    Notes
    -----
    If the aggregation is scalar/single-category (i.e. generates a 2D scalar
    mappable), the artist can be used to make a colorbar with ``fig.colorbar``.

    If the aggregation is multi-category (i.e. generates a 3D array with
    two or more components that get composited to form an image), you can use
    the :meth:`CategoricalDSArtist.get_legend_elements` method to obtain patch
    handles that can be passed to ``ax.legend`` to make a legend.

    Examples
    --------
    Generate two Gaussian point clouds and (1) plot the density as a
    quantitative map and (2) color the points by category.

    .. plot::
        :context: close-figs

        >>> import pandas as pd
        >>> import datashader as ds
        >>> import matplotlib.pyplot as plt
        >>> from datashader.mpl_ext import dsshow
        >>> n = 10000
        >>> df = pd.DataFrame({
        ...     'x': np.r_[np.random.randn(n) - 1, np.random.randn(n) + 1],
        ...     'y': np.r_[np.random.randn(n), np.random.randn(n)],
        ...     'c': pd.Categorical(np.r_[['cloud 1'] * n, ['cloud 2'] * n])
        ... })
        >>> da1 = dsshow(
        ...     df,
        ...     ds.Point('x', 'y'),
        ...     aspect='equal'
        ... )
        >>> plt.colorbar(da1);  # doctest: +SKIP
        >>> da2 = dsshow(
        ...     df,
        ...     ds.Point('x', 'y'),
        ...     ds.count_cat('c'),
        ...     aspect='equal'
        ... )
        >>> plt.legend(handles=da2.get_legend_elements());  # doctest: +SKIP

    """
    import matplotlib.pyplot as plt

    if fignum == 0:
        ax = plt.gca()
    elif ax is None:
        # Make appropriately sized figure.
        fig = plt.figure(fignum)
        ax = fig.add_axes([0.15, 0.09, 0.775, 0.775])

    if isinstance(reduction, reductions.by):
        artist = CategoricalDSArtist(
            ax,
            df,
            glyph,
            reduction,
            on_agg,
            on_color,
            color_key=color_key,
            alpha_range=alpha_range,
            color_baseline=color_baseline,
            **kwargs
        )
    else:
        if cmap is not None:
            if isinstance(cmap, list):
                cmap = mpl.colors.LinearSegmentedColormap.from_list("_datashader", cmap)

        if norm is None:
            norm = mpl.colors.Normalize()
        elif isinstance(norm, str):
            if norm == "linear":
                norm = mpl.colors.Normalize()
            elif norm == "log":
                norm = mpl.colors.LogNorm()
            elif norm == "cbrt":
                norm = mpl.colors.PowerNorm(1 / 3)
            elif norm == "eq_hist":
                norm = EqHistNormalize()

        if not isinstance(norm, mpl.colors.Normalize):
            raise ValueError(
                "`norm` must be one of 'linear', 'log', 'cbrt', 'eq_hist', "
                "or a matplotlib norm instance."
            )

        if vmin is not None:
            norm.vmin = vmin
        if vmax is not None:
            norm.vmax = vmax

        artist = ScalarDSArtist(
            ax,
            df,
            glyph,
            reduction,
            on_agg,
            on_color,
            norm=norm,
            cmap=cmap,
            alpha=alpha,
            **kwargs
        )

    ax.add_artist(artist)

    if aspect is None:
        aspect = plt.rcParams["image.aspect"]
    ax.set_aspect(aspect)

    return artist
