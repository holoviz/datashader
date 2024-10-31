from __future__ import annotations
from math import floor
import numpy as np
from toolz import memoize

from datashader.glyphs.points import _PointLike
from datashader.utils import isreal, ngjit


class _PolygonLike(_PointLike):
    """_PointLike class, with methods overridden for vertex-delimited shapes.

    Key differences from _PointLike:
        - added self.z as a list, representing vertex weights
        - constructor accepts additional kwargs:
            * weight_type (bool): Whether the weights are on vertices (True) or on the shapes
                                  (False)
            * interp (bool): Whether to interpolate (True), or to have one color per shape (False)
    """
    def __init__(self, x, y, z=None, weight_type=True, interp=True):
        super().__init__(x, y)
        if z is None:
            self.z = []
        else:
            self.z = z
        self.interpolate = interp
        self.weight_type = weight_type

    @property
    def ndims(self):
        return None

    @property
    def inputs(self):
        return (tuple([self.x, self.y] + list(self.z)) +
                (self.weight_type, self.interpolate))

    def validate(self, in_dshape):
        for col in [self.x, self.y] + list(self.z):
            if not isreal(in_dshape.measure[str(col)]):
                raise ValueError('{} must be real'.format(col))

    def required_columns(self):
        return [self.x, self.y] + list(self.z)

    def compute_x_bounds(self, df):
        xs = df[self.x].values
        bounds = self._compute_bounds(xs.reshape(np.prod(xs.shape)))
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        ys = df[self.y].values
        bounds = self._compute_bounds(ys.reshape(np.prod(ys.shape)))
        return self.maybe_expand_bounds(bounds)


class Triangles(_PolygonLike):
    """An unstructured mesh of triangles, with vertices defined by ``xs`` and ``ys``.

    Parameters
    ----------
    xs, ys, zs : list of str
        Column names of x, y, and (optional) z coordinates of each vertex.
    """
    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2,
                      _antialias_stage_2_funcs):
        draw_triangle, draw_triangle_interp = _build_draw_triangle(append)
        map_onto_pixel = _build_map_onto_pixel_for_triangle(x_mapper, y_mapper)
        extend_triangles = _build_extend_triangles(draw_triangle, draw_triangle_interp,
                                                   map_onto_pixel)
        weight_type = self.weight_type
        interpolate = self.interpolate

        def extend(aggs, df, vt, bounds, plot_start=True):
            cols = info(df, aggs[0].shape[:2])
            assert cols, 'There must be at least one column on which to aggregate'
            # mapped to pixels, then may be clipped
            extend_triangles(vt, bounds, df.values, weight_type, interpolate, aggs, cols)

        return extend


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
            append(minx, miny, *(aggs + ((w0 + w1 + w2) / 3,)))
        else:
            (ax, ay), (bx, by), (cx, cy) = verts
            bias0, bias1, bias2 = biases
            area = edge_func(ax, ay, bx, by, cx, cy)
            for j in range(miny, maxy+1):
                for i in range(minx, maxx+1):
                    g2 = edge_func(ax, ay, bx, by, i, j)
                    g0 = edge_func(bx, by, cx, cy, i, j)
                    g1 = edge_func(cx, cy, ax, ay, i, j)
                    if ((g2 > 0 or (bias0 < 0 and g2 == 0)) and
                        (g0 > 0 or (bias1 < 0 and g0 == 0)) and
                        (g1 > 0 or (bias2 < 0 and g1 == 0))):
                        interp_res = (g0 * w0 + g1 * w1 + g2 * w2) / area
                        append(i, j, *(aggs + (interp_res,)))

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
            append(minx, miny, *(aggs + (val,)))
        else:
            (ax, ay), (bx, by), (cx, cy) = verts
            bias0, bias1, bias2 = biases
            for j in range(miny, maxy+1):
                for i in range(minx, maxx+1):
                    g2 = edge_func(ax, ay, bx, by, i, j)
                    g0 = edge_func(bx, by, cx, cy, i, j)
                    g1 = edge_func(cx, cy, ax, ay, i, j)
                    if ((g2 > 0 or (bias0 < 0 and g2 == 0)) and
                        (g0 > 0 or (bias1 < 0 and g0 == 0)) and
                        (g1 > 0 or (bias2 < 0 and g1 == 0))):
                        append(i, j, *(aggs + (val,)))

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

        max_x_pixels = round((bounds[1] - bounds[0])*vt[0]) - 1
        max_y_pixels = round((bounds[3] - bounds[2])*vt[2]) - 1

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

            # Convert bbox to integer pixels
            minx = max(floor(minx+0.5), 0)
            miny = max(floor(miny+0.5), 0)
            maxx = min(floor(maxx+0.5), max_x_pixels)
            maxy = min(floor(maxy+0.5), max_y_pixels)

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


def _build_map_onto_pixel_for_triangle(x_mapper, y_mapper):
    @ngjit
    def map_onto_pixel(vt, bounds, x, y):
        """Map points onto pixel grid.
        """
        # Do not snap to pixel centers
        sx, tx, sy, ty = vt
        xx = x_mapper(x)*sx + tx - 0.5
        yy = y_mapper(y)*sy + ty - 0.5
        return xx, yy

    return map_onto_pixel
