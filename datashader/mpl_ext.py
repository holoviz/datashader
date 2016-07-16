from __future__ import absolute_import
import datashader as ds


import numpy as np


import matplotlib.image as mimage
from matplotlib.transforms import (Bbox, TransformedBbox, BboxTransform)


class DSArtist(mimage._ImageBase):
    def __init__(self, ax, df, x, y, agg=None, **kwargs):
        super().__init__(ax, **kwargs)
        self.df = df
        self.x = x
        self.y = y
        self.agg = agg
        self.axes = ax
        ax.set_ylim((df[y].min(), df[y].max()))
        ax.set_xlim((df[x].min(), df[x].max()))
        self.set_array([[1, 1], [1, 1]])

    def make_image(self, renderer, magnification=1.0,
                   unsampled=False):
        trans = self.get_transform()
        (x1, x2), (y1, y2) = self.axes.get_xlim(), self.axes.get_ylim()
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        transformed_bbox = TransformedBbox(bbox, trans)

        dims = self.axes.axesPatch.get_window_extent().bounds
        plot_width = int(dims[3] + 0.5)
        plot_height = int(dims[2] + 0.5)

        cvs = ds.Canvas(plot_width=plot_width,
                        plot_height=plot_height,
                        x_range=(x1, x2),
                        y_range=(y1, y2))
        img = cvs.points(self.df, self.x, self.y, self.agg)
        img = np.flipud(img)
        self.set_clim(np.min(img)+1, np.max(img))
        self.set_array(img)
        return self._make_image(
            img, bbox, transformed_bbox, self.axes.bbox, magnification,
            unsampled=unsampled)

    def get_extent(self):
        return (*self.axes.get_xlim(), *self.axes.get_ylim())

    def get_cursor_data(self, event):
        """Get the cursor data for a given event"""
        xmin, xmax, ymin, ymax = self.get_extent()
        if self.origin == 'upper':
            ymin, ymax = ymax, ymin
        arr = self.get_array()
        data_extent = Bbox([[ymin, xmin], [ymax, xmax]])
        array_extent = Bbox([[0, 0], arr.shape[:2]])
        trans = BboxTransform(boxin=data_extent,
                              boxout=array_extent)
        y, x = event.ydata, event.xdata
        i, j = trans.transform_point([y, x]).astype(int)
        # Clip the coordinates at array bounds
        if not (0 <= i < arr.shape[0]) or not (0 <= j < arr.shape[1]):
            return None
        else:
            return arr[i, j]
