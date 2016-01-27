from __future__ import absolute_import, division, print_function

from datashape.predicates import istabular
from odo import discover

from .utils import Dispatcher


class Axis(object):
    def __eq__(self, other):
        return (type(self) == type(other) and
                self.start == other.start and
                self.end == other.end)


class LinearAxis(Axis):
    def __init__(self, range):
        self.start, self.end = range


class LogAxis(Axis):
    def __init__(self, range):
        self.start, self.end = range


_axis_types = {'linear': LinearAxis,
               'log': LogAxis}


class Canvas(object):
    def __init__(self, plot_width=600, plot_height=600,
                 x_range=None, y_range=None,
                 x_axis_type='linear', y_axis_type='linear',
                 stretch=False):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.x_range = tuple(x_range) if x_range else x_range
        self.y_range = tuple(y_range) if y_range else y_range
        self.x_axis_type = _axis_types[x_axis_type]
        self.y_axis_type = _axis_types[y_axis_type]
        self.stretch = stretch

    def points(self, source, x, y, agg):
        from .glyphs import Point
        return bypixel(source, self, Point(x, y), agg)

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
