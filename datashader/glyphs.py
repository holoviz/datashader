from __future__ import absolute_import, division

from toolz import memoize

from .utils import ngjit, isreal
from .dispatch import dispatch


class Glyph(object):
    pass


class Point(Glyph):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def validate(self, in_dshape):
        if not isreal(in_dshape.measure[self.x]):
            raise ValueError('x must be real')
        elif not isreal(in_dshape.measure[self.y]):
            raise ValueError('y must be real')

    @property
    def inputs(self):
        return self.x, self.y

    @memoize
    def _build_extend(self, info, append):
        x_name = self.x
        y_name = self.y

        @ngjit
        def _extend(vt, xs, ys, *aggs_and_cols):
            sx, sy, tx, ty = vt
            for i in range(xs.shape[0]):
                append(i, int(xs[i] * sx + tx), int(ys[i] * sy + ty),
                       *aggs_and_cols)

        def extend(aggs, df, vt):
            xs = df[x_name].values
            ys = df[y_name].values
            cols = aggs + info(df)
            _extend(vt, xs, ys, *cols)

        return extend


@dispatch(Point, object, object)
def subselect(glyph, df, canvas):
    select = None
    if canvas.x_range:
        xmin, xmax = canvas.x_range
        x = df[glyph.x]
        select = (x >= xmin) & (x <= xmax)
    if canvas.y_range:
        ymin, ymax = canvas.y_range
        y = df[glyph.y]
        temp = (y >= ymin) & (y <= ymax)
        select = temp if select is None else temp & select
    if select is None:
        return df
    return df[select]


@dispatch(Point, object)
def compute_x_bounds(glyph, df):
    return df[glyph.x].min(), df[glyph.x].max()


@dispatch(Point, object)
def compute_y_bounds(glyph, df):
    return df[glyph.y].min(), df[glyph.y].max()
