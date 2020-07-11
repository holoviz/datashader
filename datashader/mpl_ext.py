from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform
import matplotlib as mpl
import matplotlib.image as mimage
import numpy as np

from . import transfer_functions as tf
from . import Canvas
from .core import bypixel


def uint32_to_uint8(img):
    """Cast uint32 RGB image to 4 uint8 channels."""
    return np.flipud(img.view(dtype=np.uint8).reshape(img.shape + (4,)))


def uint8_to_uint32(img):
    """Cast 4 uint8 channels into uint32 RGB image"""
    return img.view(dtype=np.uint32).reshape(img.shape[:-1])


class DSArtist(mimage._ImageBase):
    def __init__(self, ax, pipeline, extent=None, **kwargs):
        super().__init__(ax, **kwargs)
        self.pipeline = pipeline

        if extent is not None:
            ax.set_ylim(extent[0], extent[1])
            ax.set_xlim(extent[2], extent[3])
        else:
            x_col = self.pipeline.glyph.x_label
            y_col = self.pipeline.glyph.y_label
            df = self.pipeline.df
            ax.set_ylim((df[y_col].min(), df[y_col].max()))
            ax.set_xlim((df[x_col].min(), df[x_col].max()))

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
        bins = bypixel(self.pipeline.df, canvas, self.pipeline.glyph, self.pipeline.agg)
        bins = self.pipeline.transform_fn(bins)

        # infer the colormap
        vmin = bins.min()
        vmax = bins.max()
        vramp = np.linspace(vmin, vmax, 256)
        cramp = self.pipeline.color_fn(tf.Image(vramp[:, np.newaxis])).data.ravel()
        colors = uint32_to_uint8(cramp)
        cmap = mpl.colors.ListedColormap(colors / 256, name="from_datashader")
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        self.set_norm(norm)
        self.set_cmap(cmap)

        # shading part of pipeline
        img = self.pipeline.color_fn(bins)
        img = self.pipeline.spread_fn(img)
        img = uint32_to_uint8(img.data)
        img = np.ma.masked_array(img)

        # self.set_clim(vmin, vmax)
        self.set_array(img)

        return self._make_image(
            img,
            bbox,
            transformed_bbox,
            self.axes.bbox,
            magnification,
            unsampled=unsampled,
        )

    def get_extent(self):
        return (*self.axes.get_xlim(), *self.axes.get_ylim())

    def get_cursor_data(self, event):
        """Get the cursor data for a given event"""
        xmin, xmax, ymin, ymax = self.get_extent()
        if self.origin == "upper":
            ymin, ymax = ymax, ymin
        arr = self.get_array()
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
