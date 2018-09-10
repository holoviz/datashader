from __future__ import absolute_import, division

from toolz import memoize
import numpy as np

from .utils import ngjit, isreal, Expr


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
        minval = np.inf
        maxval = -np.inf
        for x in xs:
            if not np.isnan(x):
                if x < minval:
                    minval = x
                if x > maxval:
                    maxval = x

        if not (np.isfinite(minval) and np.isfinite(maxval)):
            #print("No x values; defaulting to range -0.5,0.5")
            minval, maxval = -0.5, 0.5
        elif minval==maxval:
            #print("No x range; defaulting to x-0.5,x+0.5")
            minval, maxval = minval-0.5, minval+0.5
        return minval, maxval

    @staticmethod
    @ngjit
    def _compute_y_bounds(ys):
        minval = np.inf
        maxval = -np.inf
        for y in ys:
            if not np.isnan(y):
                if y < minval:
                    minval = y
                if y > maxval:
                    maxval = y

        if not (np.isfinite(minval) and np.isfinite(maxval)):
            #print("No y values; defaulting to range -0.5,0.5")
            minval, maxval = -0.5, 0.5
        elif minval==maxval:
            #print("No y range; defaulting to y-0.5,y+0.5")
            minval, maxval = minval-0.5, minval+0.5
        return minval, maxval

    @memoize
    def _compute_x_bounds_dask(self, df):
        """Like ``PointLike._compute_x_bounds``, but memoized because
        ``df`` is immutable/hashable (a Dask dataframe).
        """
        xs = df[self.x].values
        minval, maxval = np.nanmin(xs), np.nanmax(xs)
        
        if minval == np.nan and maxval == np.nan:
            #print("No x values; defaulting to range -0.5,0.5")
            minval, maxval = -0.5, 0.5
        elif minval==maxval:
            #print("No x range; defaulting to x-0.5,x+0.5")
            minval, maxval = minval-0.5, minval+0.5
        return minval, maxval
        

    @memoize
    def _compute_y_bounds_dask(self, df):
        """Like ``PointLike._compute_y_bounds``, but memoized because
        ``df`` is immutable/hashable (a Dask dataframe).
        """
        ys = df[self.y].values
        minval, maxval = np.nanmin(ys), np.nanmax(ys)
        
        if minval == np.nan and maxval == np.nan:
            #print("No y values; defaulting to range -0.5,0.5")
            minval, maxval = -0.5, 0.5
        elif minval==maxval:
            #print("No y range; defaulting to y-0.5,y+0.5")
            minval, maxval = minval-0.5, minval+0.5
        return minval, maxval



class _PolygonLike(_PointLike):
    """_PointLike class, with methods overridden for vertex-delimited shapes.

    Key differences from _PointLike:
        - added self.z as a list, representing vertex weights
        - constructor accepts additional kwargs:
            * weight_type (bool): Whether the weights are on vertices (True) or on the shapes (False)
            * interp (bool): Whether to interpolate (True), or to have one color per shape (False)
    """
    def __init__(self, x, y, z=None, weight_type=True, interp=True):
        super(_PolygonLike, self).__init__(x, y)
        if z is None:
            self.z = []
        else:
            self.z = z
        self.interpolate = interp
        self.weight_type = weight_type

    @property
    def inputs(self):
        return tuple([self.x, self.y] + list(self.z))

    def validate(self, in_dshape):
        for col in [self.x, self.y] + list(self.z):
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
                """Map points onto pixel grid.

                Points falling on upper bound are mapped into previous bin.
                """
                xx = int(x_mapper(x) * sx + tx)
                yy = int(y_mapper(y) * sy + ty)
                return (xx - 1 if x == xmax else xx,
                        yy - 1 if y == ymax else yy)

            for i in range(xs.shape[0]):
                x = xs[i]
                y = ys[i]
                # points outside bounds are dropped; remainder
                # are mapped onto pixels
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
        draw_line = _build_draw_line(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_line = _build_extend_line(draw_line, map_onto_pixel)
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = df[x_name].values
            ys = df[y_name].values
            cols = aggs + info(df)
            # line may be clipped, then mapped to pixels
            extend_line(vt, bounds, xs, ys, plot_start, *cols)

        return extend


class Triangles(_PolygonLike):
    """An unstructured mesh of triangles, with vertices defined by ``xs`` and ``ys``.

    Parameters
    ----------
    xs, ys, zs : list of str
        Column names of x, y, and (optional) z coordinates of each vertex.
    """
    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_triangle, draw_triangle_interp = _build_draw_triangle(append)
        map_onto_pixel = _build_map_onto_pixel_for_triangle(x_mapper, y_mapper)
        extend_triangles = _build_extend_triangles(draw_triangle, draw_triangle_interp, map_onto_pixel)

        def extend(aggs, df, vt, bounds, weight_type=True, interpolate=True):
            cols = info(df)
            assert cols, 'There must be at least one column on which to aggregate'
            # mapped to pixels, then may be clipped
            extend_triangles(vt, bounds, df.values, weight_type, interpolate, aggs, cols)


        return extend


# -- Helpers for computing geometries --


def _build_map_onto_pixel_for_line(x_mapper, y_mapper):
    @ngjit
    def map_onto_pixel(vt, bounds, x, y):
        """Map points onto pixel grid.

        Points falling on upper bound are mapped into previous bin.

        If the line has been clipped, x and y will have been
        computed to lie on the bounds; we compare point and bounds
        in integer space to avoid fp error. In contrast, with
        auto-ranging, a point on the bounds will be the same
        floating point number as the bound, so comparison in fp
        representation of continuous space or in integer space
        doesn't change anything.
        """
        sx, tx, sy, ty = vt
        xmax, ymax = bounds[1], bounds[3]
        xx = int(x_mapper(x) * sx + tx)
        yy = int(y_mapper(y) * sy + ty)

        xxmax = int(x_mapper(xmax) * sx + tx)
        yymax = int(y_mapper(ymax) * sy + ty)

        return (xx - 1 if xx == xxmax else xx,
                yy - 1 if yy == yymax else yy)

    return map_onto_pixel


def _build_map_onto_pixel_for_triangle(x_mapper, y_mapper):
    @ngjit
    def map_onto_pixel(vt, bounds, x, y):
        """Map points onto pixel grid.

        Points falling on upper bound are mapped into previous bin.
        """
        sx, tx, sy, ty = vt
        xmax, ymax = bounds[1], bounds[3]
        xx = int(x_mapper(x) * sx + tx)
        yy = int(y_mapper(y) * sy + ty)
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


def _build_draw_triangle(append):
    """Specialize a triangle plotting kernel for a given append/axis combination"""
    @ngjit
    def edge_func(ax, ay, bx, by, cx, cy):
        return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax)

    @ngjit
    def draw_triangle_interp(verts, bbox, biases, aggs, weights):
        """Same as `draw_triangle()`, but with weights interpolated from vertex
        values.
        """
        minx, maxx, miny, maxy = bbox
        w0, w1, w2 = weights
        if minx == maxx and miny == maxy:
            # Subpixel case; area == 0
            append(minx, miny, aggs, (w0 + w1 + w2) / 3)
        else:
            (ax, ay), (bx, by), (cx, cy) = verts
            bias0, bias1, bias2 = biases
            area = edge_func(ax, ay, bx, by, cx, cy)
            for j in range(miny, maxy+1):
                for i in range(minx, maxx+1):
                    g2 = edge_func(ax, ay, bx, by, i, j)
                    g0 = edge_func(bx, by, cx, cy, i, j)
                    g1 = edge_func(cx, cy, ax, ay, i, j)
                    if ((g2 + bias0) | (g0 + bias1) | (g1 + bias2)) >= 0:
                        interp_res = (g0 * w0 + g1 * w1 + g2 * w2) / area
                        append(i, j, aggs, interp_res)

    @ngjit
    def draw_triangle(verts, bbox, biases, aggs, val):
        """Draw a triangle on a grid.

        Plots a triangle with integer coordinates onto a pixel grid,
        clipping to the bounds. The vertices are assumed to have
        already been scaled and transformed.
        """
        minx, maxx, miny, maxy = bbox
        if minx == maxx and miny == maxy:
            # Subpixel case; area == 0
            append(minx, miny, aggs, val)
        else:
            (ax, ay), (bx, by), (cx, cy) = verts
            bias0, bias1, bias2 = biases
            for j in range(miny, maxy+1):
                for i in range(minx, maxx+1):
                    if ((edge_func(ax, ay, bx, by, i, j) + bias0) >= 0 and
                            (edge_func(bx, by, cx, cy, i, j) + bias1) >= 0 and
                            (edge_func(cx, cy, ax, ay, i, j) + bias2) >= 0):
                        append(i, j, aggs, val)


    return draw_triangle, draw_triangle_interp


def _build_extend_triangles(draw_triangle, draw_triangle_interp, map_onto_pixel):
    @ngjit
    def extend_triangles(vt, bounds, verts, weight_type, interpolate, aggs, cols):
        """Aggregate along an array of triangles formed by arrays of CW
        vertices. Each row corresponds to a single triangle definition.

        `weight_type == True` means "weights are on vertices"
        """
        xmin, xmax, ymin, ymax = bounds
        cmax_x, cmax_y = max(xmin, xmax), max(ymin, ymax)
        cmin_x, cmin_y = min(xmin, xmax), min(ymin, ymax)
        vmax_x, vmax_y = map_onto_pixel(vt, bounds, cmax_x, cmax_y)
        vmin_x, vmin_y = map_onto_pixel(vt, bounds, cmin_x, cmin_y)

        col = cols[0] # Only aggregate over one column, for now
        n_tris = verts.shape[0]
        for n in range(0, n_tris, 3):
            a = verts[n]
            b = verts[n+1]
            c = verts[n+2]
            axn, ayn = a[0], a[1]
            bxn, byn = b[0], b[1]
            cxn, cyn = c[0], c[1]
            col0, col1, col2 = col[n], col[n+1], col[n+2]

            # Map triangle vertices onto pixels
            ax, ay = map_onto_pixel(vt, bounds, axn, ayn)
            bx, by = map_onto_pixel(vt, bounds, bxn, byn)
            cx, cy = map_onto_pixel(vt, bounds, cxn, cyn)

            # Get bounding box
            minx = min(ax, bx, cx)
            maxx = max(ax, bx, cx)
            miny = min(ay, by, cy)
            maxy = max(ay, by, cy)

            # Skip any further processing of triangles outside of viewing area
            if (minx >= vmax_x or
                    maxx < vmin_x or
                    miny >= vmax_y or
                    maxy < vmin_y):
                continue

            # Clip bbox to viewing area
            minx = max(minx, vmin_x)
            maxx = min(maxx, vmax_x)
            miny = max(miny, vmin_y)
            maxy = min(maxy, vmax_y)

            # Prevent double-drawing edges.
            # https://msdn.microsoft.com/en-us/library/windows/desktop/bb147314(v=vs.85).aspx
            bias0, bias1, bias2 = -1, -1, -1
            if ay < by or (by == ay and ax < bx):
                 bias0 = 0
            if by < cy or (cy == by and bx < cx):
                 bias1 = 0
            if cy < ay or (ay == cy and cx < ax):
                 bias2 = 0

            bbox = minx, maxx, miny, maxy
            biases = bias0, bias1, bias2
            mapped_verts = (ax, ay), (bx, by), (cx, cy)

            # draw triangles (will be clipped where outside bounds)
            if interpolate:
                weights = col0, col1, col2
                draw_triangle_interp(mapped_verts, bbox, biases, aggs, weights)
            else:
                val = (col[n] + col[n+1] + col[n+2]) / 3
                draw_triangle(mapped_verts, bbox, biases, aggs, val)

    return extend_triangles
