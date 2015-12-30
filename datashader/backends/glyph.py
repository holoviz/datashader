from __future__ import absolute_import, division

from blaze import dispatch
from toolz import memoize

from ..expr import Point
from .util import ngjit


@dispatch(Point, object, object)
@memoize
def make_extend(glyph, info, append):
    x_name = glyph.x._name
    y_name = glyph.y._name

    @ngjit
    def _extend(vt, xs, ys, *aggs_and_cols):
        sx, sy, tx, ty = vt
        for i in range(xs.shape[0]):
            append(i, int(xs[i] * sx + tx), int(ys[i] * sy + ty), *aggs_and_cols)

    def extend(aggs, df, vt):
        xs = df[x_name].values
        ys = df[y_name].values
        cols = aggs + info(df)
        _extend(vt, xs, ys, *cols)

    return extend


@dispatch(Point)
def subselect(glyph, df=None, x_range=None, y_range=None):
    xmin, xmax = x_range
    ymin, ymax = y_range
    return df[(glyph.x > xmin) & (glyph.x < xmax) &
              (glyph.y > ymin) & (glyph.y < ymax)]


def view_transform(canvas):
    w = canvas.plot_width
    h = canvas.plot_height
    xmin, xmax = canvas.x_range
    ymin, ymax = canvas.y_range
    # Compute vt
    sx = (w - 1)/(xmax - xmin)
    sy = (h - 1)/(ymax - ymin)
    if not canvas.stretch:
        sx = sy = min(sx, sy)
    tx = -xmin * sx
    ty = -ymin * sy
    return (sx, sy, tx, ty)
