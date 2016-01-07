from __future__ import absolute_import, division

from toolz import memoize

from .util import ngjit, isreal


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
