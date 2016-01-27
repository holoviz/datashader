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
        def _extend(vt, bounds, xs, ys, *aggs_and_cols):
            sx, sy, tx, ty = vt
            xmin, xmax, ymin, ymax = bounds
            for i in range(xs.shape[0]):
                x = xs[i]
                y = ys[i]
                if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                    append(i, int(x * sx + tx), int(y * sy + ty),
                           *aggs_and_cols)

        def extend(aggs, df, vt, bounds):
            xs = df[x_name].values
            ys = df[y_name].values
            cols = aggs + info(df)
            _extend(vt, bounds, xs, ys, *cols)

        return extend


@dispatch(Point, object)
def compute_x_bounds(glyph, df):
    return df[glyph.x].min(), df[glyph.x].max()


@dispatch(Point, object)
def compute_y_bounds(glyph, df):
    return df[glyph.y].min(), df[glyph.y].max()
