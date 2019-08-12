from toolz import memoize

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
        def _extend(vt, bounds, xs, ys, *aggs_and_cols):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds

            # Compute max valid x and y index
            xmaxi = int(round(x_mapper(xmax) * sx + tx))
            ymaxi = int(round(y_mapper(ymax) * sy + ty))

            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    x0, x1 = max(xs[i], xmin), min(xs[i + 1], xmax)
                    y0, y1 = max(ys[j], ymin), min(ys[j + 1], ymax)

                    # Makes sure x0 <= x1 and y0 <= y1
                    if x0 > x1:
                        x0, x1 = x1, x0
                    if y0 > y1:
                        y0, y1 = y1, y0

                    # check whether we can skip quad. To avoid overlapping
                    # quads, skip if upper bound equals viewport lower bound.
                    if x1 <= xmin or x0 > xmax or y1 <= ymin or y0 > ymax:
                        continue

                    # Map onto pixels and clip to viewport
                    x0i = max(int(x_mapper(x0) * sx + tx), 0)
                    x1i = min(int(x_mapper(x1) * sx + tx), xmaxi)
                    y0i = max(int(y_mapper(y0) * sy + ty), 0)
                    y1i = min(int(y_mapper(y1) * sy + ty), ymaxi)

                    # Make sure single pixel quads are represented
                    if x0i == x1i and x1i < ymaxi:
                        x1i += 1

                    if y0i == y1i and y1i < ymaxi:
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
            cols = aggs + info(xr_ds.transpose(y_name, x_name))
            _extend(vt, bounds, xs, ys, *cols)

        return extend


class QuadMeshCurvialinear(_QuadMeshLike):
    pass
