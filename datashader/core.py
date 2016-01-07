from __future__ import absolute_import, division, print_function

from datashape.predicates import istabular
from odo import discover

from .aggregates import Summary
from .glyphs import Point
from .util import Dispatcher, isreal


class Canvas(object):
    def __init__(self, plot_width=600, plot_height=600,
                 x_range=None, y_range=None, stretch=False):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.x_range = x_range
        self.y_range = y_range
        self.stretch = stretch

    def points(self, source, x, y, **kwargs):
        return bypixel(source, self, Point(x, y), Summary(**kwargs))

    def view_transform(self, x_range=None, y_range=None):
        w = self.plot_width
        h = self.plot_height
        xmin, xmax = x_range or self.x_range
        ymin, ymax = y_range or self.y_range
        # Compute vt
        sx = (w - 1)/(xmax - xmin)
        sy = (h - 1)/(ymax - ymin)
        if not self.stretch:
            sx = sy = min(sx, sy)
        tx = -xmin * sx
        ty = -ymin * sy
        return (sx, sy, tx, ty)


pipeline = Dispatcher()


def bypixel(source, canvas, glyph, summary):
    dshape = discover(source)
    if not istabular(dshape):
        raise ValueError("source must be tabular")
    schema = dshape.measure
    glyph.validate(schema)
    summary.validate(schema)
    return pipeline(source, schema, canvas, glyph, summary)
