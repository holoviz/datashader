from __future__ import absolute_import, division

from toolz import memoize
import numpy as np

from .core import Expr
from .utils import ngjit, isreal


class Glyph(Expr):
    """Base class for glyphs."""
    pass


class _PointLike(Glyph):
    """Shared methods between Point and Line"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def inputs(self):
        return (self.x, self.y)

    def validate(self, in_dshape):
        if not isreal(in_dshape.measure[self.x]):
            raise ValueError('x must be real')
        elif not isreal(in_dshape.measure[self.y]):
            raise ValueError('y must be real')

    def _compute_x_bounds(self, df):
        return df[self.x].min(), df[self.x].max()

    def _compute_y_bounds(self, df):
        return df[self.y].min(), df[self.y].max()


class Point(_PointLike):
    """A point, with center at ``x`` and ``y``.

    Points map each record to a single bin.

    Parameters
    ----------
    x, y : str
        Column names for the x and y coordinates of each point.
    """
    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        x_name = self.x
        y_name = self.y

        @ngjit
        def _extend(vt, bounds, xs, ys, *aggs_and_cols):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            for i in range(xs.shape[0]):
                x = xs[i]
                y = ys[i]
                if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                    append(i,
                           int(x_mapper(x) * sx + tx),
                           int(y_mapper(y) * sy + ty),
                           *aggs_and_cols)

        def extend(aggs, df, vt, bounds):
            xs = df[x_name].values
            ys = df[y_name].values
            cols = aggs + info(df)
            _extend(vt, bounds, xs, ys, *cols)

        return extend


class Line(_PointLike):
    """A line, with vertices defined by ``x`` and ``y``.

    Parameters
    ----------
    x, y : str
        Column names for the x and y coordinates of each vertex.
    """
    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_line = _build_draw_line(append, x_mapper, y_mapper)
        extend_line = _build_extend_line(draw_line)
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = df[x_name].values
            ys = df[y_name].values
            cols = aggs + info(df)
            extend_line(vt, bounds, xs, ys, plot_start, *cols)

        return extend


# -- Helpers for computing line geometry --

# Outcode constants
INSIDE = 0b0000
LEFT = 0b0001
RIGHT = 0b0010
BOTTOM = 0b0100
TOP = 0b1000


@ngjit
def _compute_outcode(x, y, xmin, xmax, ymin, ymax):
    """Outcodes for Cohen-Sutherland"""
    code = INSIDE

    if x < xmin:
        code |= LEFT
    elif x > xmax:
        code |= RIGHT
    if y < ymin:
        code |= BOTTOM
    elif y > ymax:
        code |= TOP
    return code


def _build_draw_line(append, x_mapper, y_mapper):
    """Specialize a line plotting kernel for a given append/axis combination"""
    @ngjit
    def draw_line(vt, bounds, x0, y0, x1, y1, i, plot_start, clipped,
                  *aggs_and_cols):
        """Draw a line using Bresenham's algorithm"""
        sx, tx, sy, ty = vt
        # Project to pixel space
        x0i = int(x_mapper(x0) * sx + tx)
        y0i = int(y_mapper(y0) * sy + ty)
        x1i = int(x_mapper(x1) * sx + tx)
        y1i = int(y_mapper(y1) * sy + ty)

        dx = x1i - x0i
        ix = (dx > 0) - (dx < 0)
        dx = abs(dx) * 2

        dy = y1i - y0i
        iy = (dy > 0) - (dy < 0)
        dy = abs(dy) * 2

        if plot_start:
            append(i, x0i, y0i, *aggs_and_cols)

        if dx >= dy:
            # If vertices weren't clipped and are concurrent in integer space,
            # call append and return, as the second vertex won't be hit below.
            if not clipped and not (dx | dy):
                append(i, x0i, y0i, *aggs_and_cols)
                return
            error = 2*dy - dx
            while x0i != x1i:
                if error >= 0 and (error or ix > 0):
                    error -= 2 * dx
                    y0i += iy
                error += 2 * dy
                x0i += ix
                append(i, x0i, y0i, *aggs_and_cols)
        else:
            error = 2*dx - dy
            while y0i != y1i:
                if error >= 0 and (error or iy > 0):
                    error -= 2 * dy
                    x0i += ix
                error += 2 * dx
                y0i += iy
                append(i, x0i, y0i, *aggs_and_cols)

    return draw_line


def _build_extend_line(draw_line):
    @ngjit
    def extend_line(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        """Aggregate along a line formed by ``xs`` and ``ys``"""
        sx, tx, sy, ty = vt
        xmin, xmax, ymin, ymax = bounds
        nrows = xs.shape[0]
        i = 0
        while i < nrows - 1:
            x0 = xs[i]
            y0 = ys[i]
            x1 = xs[i + 1]
            y1 = ys[i + 1]
            # If any of the coordinates are NaN, there's a discontinuity. Skip
            # the entire segment.
            if np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(y1):
                plot_start = True
                i += 2
                continue

            # Use Cohen-Sutherland to clip the segment to a bounding box
            outcode0 = _compute_outcode(x0, y0, xmin, xmax, ymin, ymax)
            outcode1 = _compute_outcode(x1, y1, xmin, xmax, ymin, ymax)

            accept = False
            clipped = False

            while True:
                if not (outcode0 | outcode1):
                    accept = True
                    break
                elif outcode0 & outcode1:
                    plot_start = True
                    break
                else:
                    clipped = True
                    outcode_out = outcode0 if outcode0 else outcode1
                    if outcode_out & TOP:
                        x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
                        y = ymax
                    elif outcode_out & BOTTOM:
                        x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
                        y = ymin
                    elif outcode_out & RIGHT:
                        y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
                        x = xmax
                    elif outcode_out & LEFT:
                        y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
                        x = xmin

                    if outcode_out == outcode0:
                        x0, y0 = x, y
                        outcode0 = _compute_outcode(x0, y0, xmin, xmax,
                                                    ymin, ymax)
                        # If x0 is clipped, we need to plot the new start
                        plot_start = True
                    else:
                        x1, y1 = x, y
                        outcode1 = _compute_outcode(x1, y1, xmin, xmax,
                                                    ymin, ymax)

            if accept:
                draw_line(vt, bounds, x0, y0, x1, y1, i, plot_start, clipped,
                          *aggs_and_cols)
                plot_start = False
            i += 1

    return extend_line
