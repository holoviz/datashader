import warnings

from matplotlib.image import _ImageBase
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform
from toolz import identity
import matplotlib as mpl
import numpy as np
import xarray as xr

from . import reductions
from . import transfer_functions as tf
from .core import bypixel, Canvas
from .pipeline import Pipeline

__all__ = ['DSArtist', 'dsshow']


def uint32_to_uint8(img):
    """Cast a uint32 RGBA image to a 4-channel uint8 image array."""
    return img.view(dtype=np.uint8).reshape(img.shape + (4,))


class DSArtist(_ImageBase):
    """
    Matplotlib artist for a datashading pipeline.

    Unlike the AxesImage artist, by default the ``origin`` for this artist
    is "lower" and interpolation is disabled.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        The axes the raster image will belong to.
    pipeline : :class:`datashader.Pipeline`
        A datashading pipeline callback.
    initial_x_range : tuple, optional
        A pair representing the initial data bounds for display on the x axis.
    initial_y_range : tuple, optional
        A pair representing the initial data bounds for display on the y axis.
    **kwargs
        Additional Artist properties.
    """

    def __init__(
        self,
        ax,
        pipeline,
        initial_x_range=None,
        initial_y_range=None,
        origin="lower",
        interpolation="none",
        **kwargs
    ):
        super().__init__(ax, origin=origin, interpolation=interpolation, **kwargs)
        self.pipeline = pipeline
        self.axes = ax

        df = self.pipeline.df
        if initial_x_range is None:
            x_col = self.pipeline.glyph.x_label
            initial_x_range = (df[x_col].min(), df[x_col].max())
        if initial_y_range is None:
            y_col = self.pipeline.glyph.y_label
            initial_y_range = (df[y_col].min(), df[y_col].max())
        ax.set_xlim(initial_x_range)
        ax.set_ylim(initial_y_range)

        # Run the pipeline on the initial bounds
        binned, raster = self._run_pipeline(initial_x_range, initial_y_range)

        # Set the image array
        rgba = uint32_to_uint8(raster.data)
        rgba = np.ma.masked_array(rgba)
        self.set_array(rgba)

        # Set the norm and cmap
        if binned.ndim == 2:
            norm, cmap = self._infer_colormap(binned)
            self.set_norm(norm)
            self.set_cmap(cmap)

    def _aggregate(self, x_range, y_range):
        dims = self.axes.patch.get_window_extent().bounds
        plot_width = int(dims[2] + 0.5)
        plot_height = int(dims[3] + 0.5)

        canvas = Canvas(
            plot_width=int(plot_width * self.pipeline.width_scale),
            plot_height=int(plot_height * self.pipeline.height_scale),
            x_range=x_range,
            y_range=y_range,
        )
        binned = bypixel(
            self.pipeline.df, canvas, self.pipeline.glyph, self.pipeline.agg
        )
        binned = self.pipeline.transform_fn(binned)

        return binned

    def _run_pipeline(self, x_range, y_range):
        # Binning part of the pipeline
        binned = self._aggregate(x_range, y_range)
        if binned.ndim not in (2, 3):
            raise ValueError(
                "Aggregated DataArray must have 2 or 3 dimensions; "
                "got array with shape {}".format(binned.shape)
            )

        # Shading part of the pipeline
        raster = self.pipeline.color_fn(binned)
        raster = self.pipeline.spread_fn(raster)

        # Save the binned data DataArray for cursor events.
        # Save the uint32 image DataArray for inspection.
        self._ds_data = binned
        self._ds_image = raster

        return binned, raster

    def _infer_color_dict(self, binned):
        # Infer the color key by sampling each categorical channel independently.
        name = binned.dims[2]
        categories = binned.coords[name].data
        n_categories = len(categories)

        # Make a row of one-hot vectors for each category (1, n, n).
        onehot = np.expand_dims(np.eye(n_categories), 0)

        # Convert to a 3D xarray so tf.shade knows what to do.
        # tf.shade generates a warning if there are singleton dims.
        onehot = xr.DataArray(onehot, dims=binned.dims).reindex({name: categories})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raster = self.pipeline.color_fn(onehot)

        # Extract categorical colors from the raster.
        rgb = uint32_to_uint8(raster.data)[0]
        color_key = [mpl.colors.to_hex(c) for c in rgb / 255.0]

        return dict(zip(categories, color_key))

    def _infer_colormap(self, binned):
        # Infer the color map by passing a linear ramp of values through the
        # color_fn.
        vmin = binned.min()
        vmax = binned.max()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        vramp = np.linspace(vmin, vmax, 256)

        # Convert to a 2D xarray to tf.shade can use it.
        # tf.shade generates a warning if there are singleton dims.
        vramp = xr.DataArray(vramp[np.newaxis, :], dims=binned.dims)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cramp = self.pipeline.color_fn(vramp)

        # Extract the colors from the raster.
        colors = uint32_to_uint8(cramp.data)[0]
        cmap = mpl.colors.ListedColormap(colors / 255.0, name="from_datashader")

        return norm, cmap

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        x1, x2, y1, y2 = self.get_extent()
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        trans = self.get_transform()
        transformed_bbox = TransformedBbox(bbox, trans)

        # Run the pipeline
        binned, raster = self._run_pipeline([x1, x2], [y1, y2])

        # Set the image array
        rgba = uint32_to_uint8(raster.data)
        rgba = np.ma.masked_array(rgba)
        self.set_array(rgba)

        # Set the norm and cmap
        if binned.ndim == 2:
            norm, cmap = self._infer_colormap(binned)
            self.set_norm(norm)
            self.set_cmap(cmap)
            # self.set_clim(vmin, vmax)

        return self._make_image(
            rgba,
            bbox,
            transformed_bbox,
            self.axes.bbox,
            magnification,
            unsampled=unsampled,
        )

    def get_ds_data(self):
        """
        Return the aggregated, pre-shaded :class:`xarray.DataArray` backing the
        bounding box currently displayed.
        """
        return self._ds_data

    def get_ds_image(self):
        """
        Return the uint32 raster image corresponding to the image currently
        displayed.

        Use :meth:`get_array` to get the equivalent matplotlib-style (M, N, 4)
        RGBA array.
        """
        return self._ds_image

    def get_legend_elements(self):
        """
        Return legend elements to display the color code for each category.
        If the datashading pipeline is quantitative, returns *None*.
        """
        x_range, y_range = self.axes.get_xlim(), self.axes.get_ylim()
        binned = self._aggregate(x_range, y_range)
        if binned.ndim != 3:
            return None
        color_dict = self._infer_color_dict(binned)
        return [
            Patch(facecolor=color, edgecolor="none", label=name)
            for name, color in color_dict.items()
        ]

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


def dsshow(
    df,
    glyph,
    agg=reductions.count(),
    transform_fn=identity,
    color_fn=tf.shade,
    spread_fn=tf.dynspread,
    width_scale=1.0,
    height_scale=1.0,
    ax=None,
    fignum=None,
    aspect=None,
    **kwargs
):
    """
    Display the output of a datashading pipeline applied to a dataframe.

    The plot will respond to changes in the bounding box being displayed (in
    data coordinates), such as pan/zoom events. Both quantitative and
    categorical datashading pipelines are supported.

    Parameters
    ----------
    df : pandas.DataFrame, dask.DataFrame
        Dataframe to apply the datashading pipeline to.
    glyph : Glyph
        The glyph to bin by.
    agg : Reduction, optional
        The reduction to compute per-pixel. Default is ``count()``.
    transform_fn : callable, optional
        A callable that takes the computed aggregate as an argument, and
        returns another aggregate. This can be used to do preprocessing before
        passing to the ``color_fn`` function.
    color_fn : callable, optional
        A callable that takes the output of ``tranform_fn``, and returns an
        ``Image`` object. Default is ``shade``.
    spread_fn : callable, optional
        A callable that takes the output of ``color_fn``, and returns another
        ``Image`` object. Default is ``dynspread``.
    height_scale: float, optional
        Factor by which to scale the image height relative to the axes
        bounding box in display space.
    width_scale: float, optional
        Factor by which to scale the image width relative to the axes
        bounding box in display space.
    ax : `matplotlib.Axes`
        Axes to draw into. If *None*, create a new figure or use ``fignum`` to
        draw into an existing figure.
    fignum : None or int or False
        If *None* and ``ax`` is *None*, create a new figure window with
        automatic numbering.
        If a nonzero integer and ``ax`` is *None*, draw into the figure with
        the given number (create it if it does not exist).
        If 0, use the current axes (or create one if it does not exist).
    aspect : {'equal', 'auto'} or float, default: ``rcParams["image.aspect"]``
        The aspect ratio of the axes.

    Other Parameters
    ----------------
    **kwargs
        All other kwargs are passed to the :class:`DSArtist`.

    Returns
    -------
    :class:`DSArtist`

    Notes
    -----
    If the datashading pipeline is categorical (i.e. generates a composited
    image from several categorical components), you can use the
    :meth:`DSArtist.get_legend_elements` method to obtain patch handles that
    can be passed to ``ax.legend`` to make a legend.

    If the pipeline is quantitative (i.e. generates a scalar mappable), the
    artist can be used to make a colorbar with ``fig.colorbar``.

    Examples
    --------
    Generate two gaussian point clouds and plot (1) the density as a
    quantitative map and (2) color the points by category.

    .. plot::
        :context: close-figs

        >>> import numpy as np
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
        ...     agg=ds.count_cat('c'),
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

    pipeline = Pipeline(
        df, glyph, agg, transform_fn, color_fn, spread_fn, width_scale, height_scale
    )
    artist = DSArtist(ax, pipeline, **kwargs)
    ax.add_artist(artist)

    if aspect is None:
        aspect = plt.rcParams["image.aspect"]
    ax.set_aspect(aspect)

    return artist
