from toolz import memoize
import numpy as np

from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit


class _QuadMeshLike(Glyph):
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    @property
    def ndims(self):
        return 2

    @property
    def inputs(self):
        return (self.x, self.y, self.name)

    def validate(self, in_dshape):
        if not isreal(in_dshape.measure[str(self.x)]):
            raise ValueError('x must be real')
        elif not isreal(in_dshape.measure[str(self.y)]):
            raise ValueError('y must be real')
        elif not isreal(in_dshape.measure[str(self.name)]):
            raise ValueError('aggregate value must be real')

    @property
    def x_label(self):
        return self.x

    @property
    def y_label(self):
        return self.y


class QuadMeshRectilinear(_QuadMeshLike):
    def _compute_bounds_from_1d_centers(self, xr_ds, dim):
        vals = xr_ds[dim].values

        # Assume dimension is sorted in ascending or descending order
        v0, v1, v_nm1, v_n = [
            vals[i] for i in [0, 1, -2, -1]
        ]

        # Check if we should swap order
        if v_n < v0:
            v0, v1, v_nm1, v_n = v_n, v_nm1, v1, v0

        bounds = (v0 - 0.5 * (v1 - v0), v_n + 0.5 * (v_n - v_nm1))
        return self.maybe_expand_bounds(bounds)

    def compute_x_bounds(self, xr_ds):
        return self._compute_bounds_from_1d_centers(xr_ds, self.x)

    def compute_y_bounds(self, xr_ds):
        return self._compute_bounds_from_1d_centers(xr_ds, self.y)

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        x_name = self.x
        y_name = self.y

        @ngjit
        @self.expand_aggs_and_cols(append)
        def _extend(xs, ys, *aggs_and_cols):
            for i in range(len(xs) - 1):
                x0i, x1i = xs[i], xs[i + 1]

                # Make sure x0 <= x1
                if x0i > x1i:
                    x0i, x1i = x1i, x0i

                # Make sure single pixel quads are represented
                if x0i == x1i:
                    x1i += 1

                for j in range(len(ys) - 1):

                    y0i, y1i = ys[j], ys[j + 1]

                    # Make sure  y0 <= y1
                    if y0i > y1i:
                        y0i, y1i = y1i, y0i

                    # Make sure single pixel quads are represented
                    if y0i == y1i:
                        y1i += 1

                    # x1i and y1i are not included in the iteration. this
                    # serves to avoid overlapping quads and it avoids the need
                    # for special handling of quads that end on exactly on the
                    # upper bound.
                    for xi in range(x0i, x1i):
                        for yi in range(y0i, y1i):
                            append(j, i, xi, yi, *aggs_and_cols)

        def extend(aggs, xr_ds, vt, bounds):
            # Convert from bin centers to interval edges
            x_breaks = infer_interval_breaks(xr_ds[x_name].values)
            y_breaks = infer_interval_breaks(xr_ds[y_name].values)

            x0, x1, y0, y1 = bounds
            xspan = x1 - x0
            yspan = y1 - y0
            xscaled = (x_mapper(x_breaks) - x0) / xspan
            yscaled = (y_mapper(y_breaks) - y0) / yspan

            xmask = np.where((xscaled >= 0) & (xscaled <= 1))
            ymask = np.where((yscaled >= 0) & (yscaled <= 1))
            xm0, xm1 = max(xmask[0].min() - 1, 0), xmask[0].max() + 1
            ym0, ym1 = max(ymask[0].min() - 1, 0), ymask[0].max() + 1

            plot_height, plot_width = aggs[0].shape[:2]

            # Downselect xs and ys and convert to int
            xs = (xscaled[xm0:xm1 + 1] * plot_width).astype(int).clip(0, plot_width)
            ys = (yscaled[ym0:ym1 + 1] * plot_height).astype(int).clip(0, plot_height)

            # For input "column", down select to valid range
            cols_full = info(xr_ds.transpose(y_name, x_name))
            cols = tuple([c[ym0:ym1, xm0:xm1] for c in cols_full])

            aggs_and_cols = aggs + cols

            _extend(xs, ys, *aggs_and_cols)

        return extend


class QuadMeshCurvialinear(_QuadMeshLike):
    def compute_x_bounds(self, xr_ds):
        xs = xr_ds[self.x].values
        xs = infer_interval_breaks(xs, axis=1)
        xs = infer_interval_breaks(xs, axis=0)
        bounds = Glyph._compute_bounds_2d(xs)
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, xr_ds):
        ys = xr_ds[self.y].values
        ys = infer_interval_breaks(ys, axis=1)
        ys = infer_interval_breaks(ys, axis=0)
        bounds = Glyph._compute_bounds_2d(ys)
        return self.maybe_expand_bounds(bounds)

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        x_name = self.x
        y_name = self.y

        @ngjit
        @self.expand_aggs_and_cols(append)
        def _extend(plot_height, plot_width, xs, ys, *aggs_and_cols):
            y_len, x_len, = xs.shape

            for i in range(x_len - 1):
                for j in range(y_len - 1):

                    # Extract quad vertices
                    x1 = xs[j, i]
                    x2 = xs[j, i + 1]
                    x3 = xs[j + 1, i + 1]
                    x4 = xs[j + 1, i]

                    y1 = ys[j, i]
                    y2 = ys[j, i + 1]
                    y3 = ys[j + 1, i + 1]
                    y4 = ys[j + 1, i]

                    # Compute the rectilinear bounding box around the quad
                    xmin = max(min(x1, x2, x3, x4), 0)
                    xmax = min(max(x1, x2, x3, x4), plot_width - 1)
                    ymin = max(min(y1, y2, y3, y4), 0)
                    ymax = min(max(y1, y2, y3, y4), plot_height - 1)

                    # Make sure single pixel quads are represented
                    if xmin == xmax:
                        xmax += 1

                    if ymin == ymax:
                        ymax += 1

                    in_quad = []
                    for xi in range(xmin, xmax):
                        for yi in range(ymin, ymax):
                            if point_in_quad(
                                    x1, x2, x3, x4, y1, y2, y3, y4, xi, yi):
                                append(j, i, xi, yi, *aggs_and_cols)
                                in_quad.append((xi, y1))

        def extend(aggs, xr_ds, vt, bounds):
            # Convert from bin centers to interval edges
            x_breaks = xr_ds[x_name].values
            x_breaks = infer_interval_breaks(x_breaks, axis=1)
            x_breaks = infer_interval_breaks(x_breaks, axis=0)

            y_breaks = xr_ds[y_name].values
            y_breaks = infer_interval_breaks(y_breaks, axis=1)
            y_breaks = infer_interval_breaks(y_breaks, axis=0)

            # Scale x and y vertices into integer canvas coordinates
            x0, x1, y0, y1 = bounds
            xspan = x1 - x0
            yspan = y1 - y0
            xscaled = (x_mapper(x_breaks) - x0) / xspan
            yscaled = (y_mapper(y_breaks) - y0) / yspan

            plot_height, plot_width = aggs[0].shape[:2]

            xs = (xscaled * plot_width).astype(int)
            ys = (yscaled * plot_height).astype(int)

            # Question: Should we try to compute a slice of xs and ys that
            # eliminates rows and columns of quads that are all outside the
            # viewport?

            aggs_and_cols = aggs + info(xr_ds)
            _extend(plot_height, plot_width, xs, ys, *aggs_and_cols)

        return extend


@ngjit
def tri_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) +
                x2 * (y3 - y1) +
                x3 * (y1 - y2)) / 2.0)


@ngjit
def point_in_quad(x1, x2, x3, x4, y1, y2, y3, y4, x, y):
    quad_area = (tri_area(x1, y1, x2, y2, x3, y3) +
                 tri_area(x1, y1, x4, y4, x3, y3))

    area_1 = tri_area(x, y, x1, y1, x2, y2)
    area_2 = tri_area(x, y, x2, y2, x3, y3)
    area_3 = tri_area(x, y, x3, y3, x4, y4)
    area_4 = tri_area(x, y, x1, y1, x4, y4)

    return quad_area == (area_1 + area_2 + area_3 + area_4)


@ngjit
def pixel_in_quad(x1, x2, x3, x4, y1, y2, y3, y4, x0, y0):
    return (point_in_quad(x1, x2, x3, x4, y1, y2, y3, y4, x0, y0) |
            point_in_quad(x1, x2, x3, x4, y1, y2, y3, y4, x0+1, y0) |
            point_in_quad(x1, x2, x3, x4, y1, y2, y3, y4, x0, y0+1) |
            point_in_quad(x1, x2, x3, x4, y1, y2, y3, y4, x0+1, y0+1))
