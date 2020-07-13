from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform
import matplotlib as mpl
import matplotlib.image as mimage
import numpy as np

from . import transfer_functions as tf
from . import Canvas
from .core import bypixel


def uint32_to_uint8(img):
    """Cast uint32 RGB image to 4 uint8 channels."""
    return img.view(dtype=np.uint8).reshape(img.shape + (4,))


class DSArtist(mimage._ImageBase):
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

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        trans = self.get_transform()
        (x1, x2), (y1, y2) = self.axes.get_xlim(), self.axes.get_ylim()
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        transformed_bbox = TransformedBbox(bbox, trans)

        dims = self.axes.patch.get_window_extent().bounds
        plot_width = int(dims[3] + 0.5)
        plot_height = int(dims[2] + 0.5)

        # binning part of pipeline
        canvas = Canvas(
            plot_width=int(plot_width * self.pipeline.width_scale),
            plot_height=int(plot_height * self.pipeline.height_scale),
            x_range=(x1, x2),
            y_range=(y1, y2),
        )
        binned = bypixel(
            self.pipeline.df, canvas, self.pipeline.glyph, self.pipeline.agg
        )
        binned = self.pipeline.transform_fn(binned)

        # save the binned data for cursor events
        self._ds_data = binned.data

        # infer the colormap or legend depending on aggregation
        if len(bins.shape) == 3:
            cdata = np.eye(bins.shape[-1])[:, np.newaxis]
            agg_name, agg_index = list(bins.indexes.items())[-1]
            cpalette = self.pipeline.color_fn(
                tf.Image(cdata, dims=bins.dims).reindex({agg_name: agg_index})
            )
            colors = uint32_to_uint8(cpalette.data.ravel())
            colors[:, -1] = 255  # set alpha to max
            for lab, col in zip(agg_index.values, colors):
                # TODO: remove colormap and use categorical legend
                pass

        else:
            vmin = bins.min()
            vmax = bins.max()
            vramp = np.linspace(vmin, vmax, 256)
            cramp = da.pipeline.color_fn(tf.Image(vramp[:, np.newaxis])).data.ravel()
            colors = uint32_to_uint8(cramp)
            cmap = mpl.colors.ListedColormap(colors / 256, name="from_datashader")
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            da.set_norm(norm)
            da.set_cmap(cmap)

        # shading part of pipeline
        img = self.pipeline.color_fn(binned)
        img = self.pipeline.spread_fn(img)

        # save the uint32 image DataArray for inspection
        self._ds_image = img

        rgba = uint32_to_uint8(img.data)
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

    def get_extent(self):
        (x1, x2), (y1, y2) = self.axes.get_xlim(), self.axes.get_ylim()
        return x1, x2, y1, y2

    def get_cursor_data(self, event):
        """Get the cursor data for a given event"""
        xmin, xmax, ymin, ymax = self.get_extent()
        if self.origin == "upper":
            ymin, ymax = ymax, ymin

        arr = self._ds_data
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
