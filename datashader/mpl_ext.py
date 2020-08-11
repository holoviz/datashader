from matplotlib.image import _ImageBase
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform
import matplotlib as mpl
import numpy as np
import xarray as xr

from . import transfer_functions as tf
from . import Canvas
from .core import bypixel


def uint32_to_uint8(img):
    """Cast uint32 RGB image to 4 uint8 channels."""
    return img.view(dtype=np.uint8).reshape(img.shape + (4,))


class DSArtist(_ImageBase):
    def __init__(
        self,
        ax,
        pipeline,
        initial_x_range=None,
        initial_y_range=None,
        origin="lower",
        **kwargs
    ):
        super().__init__(ax, origin=origin, **kwargs)
        self.pipeline = pipeline
        df = self.pipeline.df

        if initial_x_range is not None:
            ax.set_xlim(initial_x_range)
        else:
            x_col = self.pipeline.glyph.x_label
            ax.set_xlim((df[x_col].min(), df[x_col].max()))

        if initial_y_range is not None:
            ax.set_ylim(initial_y_range)
        else:
            y_col = self.pipeline.glyph.y_label
            ax.set_ylim((df[y_col].min(), df[y_col].max()))

        self.axes = ax
        self.set_array([[1, 1], [1, 1]])

    def _aggregate(self, x_range, y_range):
        dims = self.axes.patch.get_window_extent().bounds
        plot_width = int(dims[3] + 0.5)
        plot_height = int(dims[2] + 0.5)

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

    def _infer_color_dict(self, binned):
        # Infer the color key by sampling each categorical channel independently.
        name = binned.dims[2]
        categories = binned.coords[name].data
        n_categories = len(categories)

        # Make a row of one-hot vectors for each category (1, n, n).
        # tf.shade generates a warning if there are singleton dims, so we pad
        # with an extra row (2, n, n).
        onehot = np.expand_dims(np.eye(n_categories), 0)
        onehot = np.concatenate([onehot, onehot], axis=0)

        # Convert to xarray so tf.shade knows what to do.
        onehot = xr.DataArray(
            onehot, dims=binned.dims
        ).reindex({name: categories})

        # Extract categorical colors from the raster.
        raster = self.pipeline.color_fn(onehot)
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
        vramp = xr.DataArray(vramp[:, np.newaxis], dims=binned.dims)
        cramp = self.pipeline.color_fn(vramp)

        # Extract the colors from the raster.
        colors = uint32_to_uint8(cramp.data.ravel())
        cmap = mpl.colors.ListedColormap(colors / 255.0, name="from_datashader")

        return norm, cmap

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        x1, x2, y1, y2 = self.get_extent()
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        trans = self.get_transform()
        transformed_bbox = TransformedBbox(bbox, trans)

        dims = self.axes.patch.get_window_extent().bounds
        plot_width = int(dims[3] + 0.5)
        plot_height = int(dims[2] + 0.5)

        # binning part of pipeline
        binned = self._aggregate([x1, x2], [y1, y2])
        if binned.ndim == 2:
            norm, cmap = self._infer_colormap(binned)
            self.set_norm(norm)
            self.set_cmap(cmap)
        elif binned.ndim != 3:
            raise ValueError(
                "Aggregated DataArray must have 2 or 3 dimensions; "
                "got array with shape {}".format(binned.shape)
            )
        # shading part of pipeline
        raster = self.pipeline.color_fn(binned)
        raster = self.pipeline.spread_fn(raster)

        # save the binned data DataArray for cursor events
        # save the uint32 image DataArray for inspection
        self._ds_data = binned
        self._ds_image = raster

        rgba = uint32_to_uint8(raster.data)
        rgba = np.ma.masked_array(rgba)
        # self.set_clim(vmin, vmax)
        self.set_array(rgba)

        return self._make_image(
            rgba,
            bbox,
            transformed_bbox,
            self.axes.bbox,
            magnification,
            unsampled=unsampled,
        )

    def get_legend_elements(self):
        x_range, y_range = self.axes.get_xlim(), self.axes.get_ylim()
        binned = self._aggregate(x_range, y_range)
        if binned.ndim != 3:
            return None
        color_dict = self._infer_color_dict(binned)
        return [
            Patch(facecolor=color, edgecolor='none', label=name)
            for name, color in color_dict.items()
        ]

    def get_extent(self):
        (x1, x2), (y1, y2) = self.axes.get_xlim(), self.axes.get_ylim()
        return x1, x2, y1, y2

    def get_cursor_data(self, event):
        """Get the cursor data for a given event"""
        xmin, xmax, ymin, ymax = self.get_extent()
        if self.origin == "upper":
            ymin, ymax = ymax, ymin

        arr = self._ds_data.data
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
