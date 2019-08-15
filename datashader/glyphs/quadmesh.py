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
        def _extend(vt, bounds, xs, ys, *aggs_and_cols):
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
            xs = infer_interval_breaks(xr_ds[x_name].values)
            ys = infer_interval_breaks(xr_ds[y_name].values)

            x0, x1, y0, y1 = bounds
            xspan = x1 - x0
            yspan = y1 - y0
            xscaled = (x_mapper(xs) - x0) / xspan
            yscaled = (y_mapper(ys) - y0) / yspan

            xmask = np.where((xscaled >= 0) & (xscaled <= 1))
            ymask = np.where((yscaled >= 0) & (yscaled <= 1))
            xm0, xm1 = max(xmask[0].min() - 1, 0), xmask[0].max() + 1
            ym0, ym1 = max(ymask[0].min() - 1, 0), ymask[0].max() + 1

            plot_height, plot_width = aggs[0].shape[:2]

            # Downselect xs and ys and convert to int
            xs = (xscaled[xm0:xm1 + 1] * plot_width).astype(int).clip(0, plot_width)
            ys = (yscaled[ym0:ym1 + 1] * plot_height).astype(int).clip(0, plot_height)

            # For each of aggs and cols, down select to valid range
            cols_full = info(xr_ds.transpose(y_name, x_name))
            cols = tuple([c[ym0:ym1, xm0:xm1] for c in cols_full])

            aggs_and_cols = aggs + cols

            _extend(vt, bounds, xs, ys, *aggs_and_cols)

        return extend


class QuadMeshCurvialinear(_QuadMeshLike):
    pass
