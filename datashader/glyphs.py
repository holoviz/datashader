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
        _extend = _build_line_kernel(append, x_mapper, y_mapper)
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds):
            xs = df[x_name].values
            ys = df[y_name].values
            cols = aggs + info(df)
            _extend(vt, bounds, xs, ys, *cols)

        return extend


# -- Helpers for drawing computing geometry --

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


def _build_line_kernel(append, x_mapper, y_mapper):
    """Specialize a line plotting kernel for a given append/axis combination"""
    @ngjit
    def draw_line(vt, bounds, x0, y0, x1, y1, lastx, lasty, i, *aggs_and_cols):
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

        if lastx != x0i or lasty != y0i:
            append(i, x0i, y0i, *aggs_and_cols)
        elif lastx == x1i and lasty == y1i:
            return (lastx, lasty)

        if dx >= dy:
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
        return (x0i, y0i)

    @ngjit
    def extend_lines(vt, bounds, xs, ys, *aggs_and_cols):
        """Aggregate along a line formed by ``xs`` and ``ys``"""
        sx, tx, sy, ty = vt
        xmin, xmax, ymin, ymax = bounds
        # These track the last pixel coordinate appended to, allowing us to
        # debounce on duplicate pixels.
        lastx = lasty = -1
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
                i += 2
                lastx = lasty = -1
                continue

            # Use Cohen-Sutherland to clip the segment to a bounding box
            # This is pretty much taken verbatim from Wikipedia:
            # https://en.wikipedia.org/wiki/Cohen%E2%80%93Sutherland_algorithm
            outcode0 = _compute_outcode(x0, y0, xmin, xmax, ymin, ymax)
            outcode1 = _compute_outcode(x1, y1, xmin, xmax, ymin, ymax)

            accept = False

            while True:
                if not (outcode0 | outcode1):
                    accept = True
                    break
                elif outcode0 & outcode1:
                    break
                else:
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
                    else:
                        x1, y1 = x, y
                        outcode1 = _compute_outcode(x1, y1, xmin, xmax,
                                                    ymin, ymax)

            if accept:
                lastx, lasty = draw_line(vt, bounds, x0, y0, x1, y1, lastx,
                                         lasty, i, *aggs_and_cols)
            i += 1

    return extend_lines
