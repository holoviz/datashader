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

    @staticmethod
    @ngjit
    def _compute_x_bounds(xs):
        minval = maxval = xs[0]
        for x in xs:
            if not np.isnan(x):
                if np.isnan(minval) or x < minval:
                    minval = x
                if np.isnan(maxval) or x > maxval:
                    maxval = x
        if np.isnan(minval) or np.isnan(maxval):
            raise ValueError('All x coordinates are NaN.')
        return minval, maxval

    @staticmethod
    @ngjit
    def _compute_y_bounds(ys):
        minval = maxval = ys[0]
        for y in ys:
            if not np.isnan(y):
                if np.isnan(minval) or y < minval:
                    minval = y
                if np.isnan(maxval) or y > maxval:
                    maxval = y
        if np.isnan(minval) or np.isnan(maxval):
            raise ValueError('All y coordinates are NaN.')
        return minval, maxval

    @memoize
    def _compute_x_bounds_dask(self, df):
        """Like ``PointLike._compute_x_bounds``, but memoized because
        ``df`` is immutable/hashable (a Dask dataframe).
        """
        xs = df[self.x].values
        return np.nanmin(xs), np.nanmax(xs)

    @memoize
    def _compute_y_bounds_dask(self, df):
        """Like ``PointLike._compute_y_bounds``, but memoized because
        ``df`` is immutable/hashable (a Dask dataframe).
        """
        ys = df[self.y].values
        return np.nanmin(ys), np.nanmax(ys)


class Point(_PointLike):
    """A point, with center at ``x`` and ``y``.

    Points map each record to a single bin.
    Points falling exactly on the upper bounds are treated as a special case,
    mapping into the previous bin rather than being cropped off.

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

            def map_onto_pixel(x, y):
                xx = int(x_mapper(x) * sx + tx)
                yy = int(y_mapper(y) * sy + ty)
                # Points falling on upper bound are mapped into previous bin
                return (xx - 1 if x == xmax else xx,
                        yy - 1 if y == ymax else yy)

            for i in range(xs.shape[0]):
                x = xs[i]
                y = ys[i]
                if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                    xi, yi = map_onto_pixel(x, y)
                    append(i, xi, yi, *aggs_and_cols)

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
        map_onto_pixel = _build_map_onto_pixel(x_mapper, y_mapper)
        draw_line = _build_draw_line(append)
        extend_line = _build_extend_line(draw_line, map_onto_pixel)
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = df[x_name].values
            ys = df[y_name].values
            cols = aggs + info(df)
            extend_line(vt, bounds, xs, ys, plot_start, *cols)

        return extend


# -- Helpers for computing line geometry --

def _build_map_onto_pixel(x_mapper, y_mapper):
    @ngjit
    def map_onto_pixel(vt, bounds, x, y):
        """Map points onto pixel grid"""
        sx, tx, sy, ty = vt
        _, xmax, _, ymax = bounds
        xx = int(x_mapper(x) * sx + tx)
        yy = int(y_mapper(y) * sy + ty)
        # Points falling on upper bound are mapped into previous bin
        return (xx - 1 if x == xmax else xx,
                yy - 1 if y == ymax else yy)

    return map_onto_pixel


def _build_draw_line(append):
    """Specialize a line plotting kernel for a given append/axis combination"""
    @ngjit
    def draw_line(x0i, y0i, x1i, y1i, i, plot_start, clipped, *aggs_and_cols):
        """Draw a line using Bresenham's algorithm

        This method plots a line segment with integer coordinates onto a pixel
        grid. The vertices are assumed to have already been scaled, transformed,
        and clipped within the bounds.

        The following algorithm is the more general Bresenham's algorithm that
        works with both float and integer coordinates. A future performance
        improvement would replace this algorithm with the integer-specific one.
        """
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


def _build_extend_line(draw_line, map_onto_pixel):
    @ngjit
    def outside_bounds(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
        if x0 < xmin and x1 < xmin:
            return True
        if x0 > xmax and x1 > xmax:
            return True
        if y0 < ymin and y1 < ymin:
            return True
        return y0 > ymax and y1 > ymax

    @ngjit
    def clipt(p, q, t0, t1):
        accept = True
        if p < 0 and q < 0:
            r = q / p
            if r > t1:
                accept = False
            elif r > t0:
                t0 = r
        elif p > 0 and q < p:
            r = q / p
            if r < t0:
                accept = False
            elif r < t1:
                t1 = r
        elif q < 0:
            accept = False
        return t0, t1, accept

    @ngjit
    def extend_line(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        """Aggregate along a line formed by ``xs`` and ``ys``"""
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
                i += 1
                continue

            # Use Liang-Barsky (1992) to clip the segment to a bounding box
            if outside_bounds(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
                plot_start = True
                i += 1
                continue

            clipped = False

            t0, t1 = 0, 1
            dx = x1 - x0

            t0, t1, accept = clipt(-dx, x0 - xmin, t0, t1)
            if not accept:
                i += 1
                continue

            t0, t1, accept = clipt(dx, xmax - x0, t0, t1)
            if not accept:
                i += 1
                continue

            dy = y1 - y0

            t0, t1, accept = clipt(-dy, y0 - ymin, t0, t1)
            if not accept:
                i += 1
                continue

            t0, t1, accept = clipt(dy, ymax - y0, t0, t1)
            if not accept:
                i += 1
                continue

            if t1 < 1:
                clipped = True
                x1 = x0 + t1 * dx
                y1 = y0 + t1 * dy

            if t0 > 0:
                # If x0 is clipped, we need to plot the new start
                clipped = True
                plot_start = True
                x0 = x0 + t0 * dx
                y0 = y0 + t0 * dy

            x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
            x1i, y1i = map_onto_pixel(vt, bounds, x1, y1)
            draw_line(x0i, y0i, x1i, y1i, i, plot_start, clipped, *aggs_and_cols)
            plot_start = False
            i += 1

    return extend_line
