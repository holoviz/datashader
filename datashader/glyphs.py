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


class _PolygonLike(_PointLike):
    """_PointLike class, with Triangle-specific methods overriden.

    Key differences from _PointLike:
        - self.x and self.y are tuples of strs, instead of just strs
        - self.xs and self.ys are available, as aliases to self.x and self.y
    """
    @property
    def xs(self):
        return self.x

    @property
    def ys(self):
        return self.y

    @property
    def inputs(self):
        return tuple(zip(self.xs, self.ys))

    def validate(self, in_dshape):
        for col in (self.xs + self.ys):
            if not isreal(in_dshape.measure[col]):
                raise ValueError('{} must be real'.format(col))


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


class Triangles(_PolygonLike):
    """An unstructured mesh of triangles, with vertices defined by ``xs`` and ``ys``.

    Parameters
    ----------
    xs, ys : list of str
        Column names of x and y coordinates of each vertex.
    """
    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        map_onto_pixel = _build_map_onto_pixel(x_mapper, y_mapper)
        draw_triangle = _build_draw_triangle(append)
        extend_triangles = _build_extend_triangles(draw_triangle, map_onto_pixel)
        x_names = self.xs
        y_names = self.ys

        def extend(aggs, df, vt, bounds):
            verts = df[x_names + y_names].values
            cols = aggs + info(df)
            extend_triangles(vt, bounds, verts, *cols)

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


@ngjit
def edge_func(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

def _build_draw_triangle(append):
    """Specialize a triangle plotting kernel for a given append/axis combination"""
    @ngjit
    def draw_triangle(n, verts, bbox, biases, *aggs_and_cols):
        """Draw a triangle on a grid.

        This method plots a triangle with integer coordinates onto a pixel
        grid. The vertices are assumed to have already been scaled, transformed,
        and clipped within the bounds.
        """
        (ax, ay), (bx, by), (cx, cy) = a, b, c = verts
        bias0, bias1, bias2 = biases
        minx, maxx, miny, maxy = bbox
        agg = aggs_and_cols[0]
        for i in range(minx, maxx+1):
            for j in range(miny, maxy+1):
                pt = (i, j)
                inside_tri = (edge_func(a, b, pt) + bias0) >= 0 and \
                             (edge_func(b, c, pt) + bias1) >= 0 and \
                             (edge_func(c, a, pt) + bias2) >= 0
                if inside_tri:
                    append(n, i, j, *aggs_and_cols)


    return draw_triangle


def _build_extend_triangles(draw_triangle, map_onto_pixel):
    @ngjit
    def extend_triangles(vt, bounds, verts, *aggs_and_cols):
        """Aggregate along an array of triangles formed by arrays of CW
        vertices. Each row corresponds to a single triangle definition.
        """
        xmin, xmax, ymin, ymax = bounds
        mmax_x, mmax_y = map_onto_pixel(vt, bounds, xmax, ymax)
        mmin_x, mmin_y = map_onto_pixel(vt, bounds, xmin, ymin)
        n_tris = verts.shape[0]
        for n in range(n_tris):
            aix, bix, cix, aiy, biy, ciy = verts[n]

            # Map triangle vertices onto pixels
            ax, ay = map_onto_pixel(vt, bounds, aix, aiy)
            bx, by = map_onto_pixel(vt, bounds, bix, biy)
            cx, cy = map_onto_pixel(vt, bounds, cix, ciy)

            # Prevent double-drawing edges.
            # https://msdn.microsoft.com/en-us/library/windows/desktop/bb147314(v=vs.85).aspx
            # Always draw edges of the last triangle.
            if n < (n_verts-1):
                bias0, bias1, bias2 = -1, -1, -1
                if by > ay or bx < ax:
                    bias0 = 0
                if cy > by or cx < bx:
                    bias1 = 0
                if ay > cy or ax < cx:
                    bias2 = 0
            else:
                bias0, bias1, bias2 = 0, 0, 0

            # Get bounding box
            minx = min(ax, bx, cx)
            maxx = max(ax, bx, cx)
            miny = min(ay, by, cy)
            maxy = max(ay, by, cy)

            # Clip to viewing area
            minx = max(minx, mmin_x)
            maxx = min(maxx, mmax_x)
            miny = max(miny, mmin_y)
            maxy = min(maxy, mmax_y)

            mapped_verts = (ax, ay), (bx, by), (cx, cy)
            bbox = minx, maxx, miny, maxy
            biases = bias0, bias1, bias2
            draw_triangle(n, mapped_verts, bbox, biases, *aggs_and_cols)

    return extend_triangles
