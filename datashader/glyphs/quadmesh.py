from toolz import memoize
import numpy as np

from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit
import numba
from numba import cuda

try:
    import cupy
    from datashader.transfer_functions._cuda_utils import cuda_args
except Exception:
    cupy = None
    cuda_args = None


def _cuda_mapper(mapper):
    @cuda.jit
    def kernel(in_array, out_array):
        i, j = cuda.grid(2)
        if i < out_array.shape[0] and j < out_array.shape[1]:
            out_array[i, j] = mapper(in_array[i, j])

    def cuda_map(in_array):
        out_array = cupy.zeros(in_array.shape, dtype='float64')
        in_array = cuda.to_device(in_array)
        kernel[cuda_args(in_array.shape)](in_array, out_array)
        return out_array

    return cuda_map


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
        name = self.name

        @ngjit
        @self.expand_aggs_and_cols(append)
        def perform_extend(i, j, xs, ys, *aggs_and_cols):
            x0i, x1i = xs[i], xs[i + 1]
            # Make sure x0 <= x1
            if x0i > x1i:
                x0i, x1i = x1i, x0i
            # Make sure single pixel quads are represented
            if x0i == x1i:
                x1i += 1
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

        @cuda.jit
        @self.expand_aggs_and_cols(append)
        def extend_cuda(xs, ys, *aggs_and_cols):
            i, j = cuda.grid(2)
            if i < (xs.shape[0] - 1) and j < (ys.shape[0] - 1):
                perform_extend(i, j, xs, ys, *aggs_and_cols)

        @ngjit
        @self.expand_aggs_and_cols(append)
        def extend_cpu(xs, ys, *aggs_and_cols):
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    perform_extend(i, j, xs, ys, *aggs_and_cols)

        def extend(aggs, xr_ds, vt, bounds):
            from datashader.core import LinearAxis
            use_cuda = cupy and isinstance(xr_ds[name].data, cupy.ndarray)

            xs = xr_ds[x_name].values
            ys = xr_ds[y_name].values
            if use_cuda:
                xs = cupy.array(xs)
                ys = cupy.array(ys)

                x_mapper2 = _cuda_mapper(x_mapper)
                y_mapper2 = _cuda_mapper(y_mapper)
            else:
                x_mapper2 = x_mapper
                y_mapper2 = y_mapper

            # Convert from bin centers to interval edges
            x_breaks = infer_interval_breaks(xs)
            y_breaks = infer_interval_breaks(ys)

            x0, x1, y0, y1 = bounds
            xspan = x1 - x0
            yspan = y1 - y0

            if x_mapper is LinearAxis.mapper:
                xscaled = (x_breaks - x0) / xspan
            else:
                xscaled = (x_mapper2(x_breaks) - x0) / xspan

            if y_mapper is LinearAxis.mapper:
                yscaled = (y_breaks - y0) / yspan
            else:
                yscaled = (y_mapper2(y_breaks) - y0) / yspan

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

            if use_cuda:
                do_extend = extend_cuda[cuda_args(xr_ds[name].shape)]
            else:
                do_extend = extend_cpu

            do_extend(xs, ys, *aggs_and_cols)

        return extend


class QuadMeshCurvilinear(_QuadMeshLike):
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
        name = self.name

        @ngjit
        @self.expand_aggs_and_cols(append)
        def perform_extend(
                i, j,
                plot_height, plot_width, xs, ys, xverts, yverts,
                yincreasing, eligible, intersect, *aggs_and_cols
        ):

            # make array of quad x any vertices
            xverts[0] = xs[j, i]
            xverts[1] = xs[j, i + 1]
            xverts[2] = xs[j + 1, i + 1]
            xverts[3] = xs[j + 1, i]
            xverts[4] = xverts[0]

            yverts[0] = ys[j, i]
            yverts[1] = ys[j, i + 1]
            yverts[2] = ys[j + 1, i + 1]
            yverts[3] = ys[j + 1, i]
            yverts[4] = yverts[0]

            # Compute the rectilinear bounding box around the quad and
            # skip quad if there is no chance for it to intersect
            # viewport
            xmin = min(min(xverts[0], xverts[1]), min(xverts[2], xverts[3]))
            if xmin >= plot_width:
                return
            xmin = max(xmin, 0)

            xmax = max(max(xverts[0], xverts[1]), max(xverts[2], xverts[3]))
            if xmax < 0:
                return
            xmax = min(xmax, plot_width)

            ymin = min(min(yverts[0], yverts[1]), min(yverts[2], yverts[3]))
            if ymin >= plot_height:
                return
            ymin = max(ymin, 0)

            ymax = max(max(yverts[0], yverts[1]), max(yverts[2], yverts[3]))
            if ymax < 0:
                return
            ymax = min(ymax, plot_height)

            # Handle subpixel quads
            if xmin == xmax or ymin == ymax:
                # If either dimension is a single pixel, then render it
                if xmin == xmax and xmax < plot_width:
                    xmax += 1
                if ymin == ymax and ymax < plot_height:
                    ymax += 1

                for yi in range(ymin, ymax):
                    for xi in range(xmin, xmax):
                        append(j, i, xi, yi, *aggs_and_cols)

                return

            # make yincreasing an array holding whether each edge is
            # increasing vertically (+1), decreasing vertically (-1),
            # or horizontal (0).
            yincreasing[:] = 0
            for k in range(4):
                if yverts[k + 1] > yverts[k]:
                    yincreasing[k] = 1
                elif yverts[k + 1] < yverts[k]:
                    yincreasing[k] = -1

            for yi in range(ymin, ymax):
                eligible[:] = 1
                for xi in range(xmin, xmax):
                    intersect[:] = 0
                    # Test edges
                    for edge_i in range(4):
                        # Skip if we already know edge is ineligible
                        if not eligible[edge_i]:
                            continue

                        # Check if edge is fully to left of point. If
                        # so, we don't need to consider it again for
                        # this row.
                        if ((xverts[edge_i] < xi) and
                                (xverts[edge_i + 1] < xi)):
                            eligible[edge_i] = 0
                            continue

                        # Check if edge is fully above or below point.
                        # If so, we don't need to consider it again
                        # for this  row.
                        if ((yverts[edge_i] > yi) ==
                                (yverts[edge_i + 1] > yi)):
                            eligible[edge_i] = 0
                            continue

                        # Now check if edge is to the right of point.
                        # A is vector from point to first vertex
                        ax = xverts[edge_i] - xi
                        ay = yverts[edge_i] - yi

                        # B is vector from point to second vertex
                        bx = xverts[edge_i + 1] - xi
                        by = yverts[edge_i + 1] - yi

                        # Compute cross product of B and A
                        bxa = (bx * ay - by * ax)

                        # If cross product has same sign as yincreasing
                        # then edge intersects to the right
                        intersect[edge_i] = (
                                bxa * yincreasing[edge_i] < 0
                        )
                    intersections = (
                            intersect[0] + intersect[1] + intersect[2] + intersect[3]
                    )
                    if intersections % 2 == 1:
                        # If odd number of intersections, point
                        # is inside quad
                        append(j, i, xi, yi, *aggs_and_cols)

        @cuda.jit
        @self.expand_aggs_and_cols(append)
        def extend_cuda(plot_height, plot_width, xs, ys, *aggs_and_cols):
            # # For consistency with CPU path, we initialize all arrays here
            # # xverts/yverts arrays
            xverts = cuda.local.array(5, dtype=numba.types.int32)
            yverts = cuda.local.array(5, dtype=numba.types.int32)
            #
            # # Array holding whether each edge is increasing
            # # vertically (+1), decreasing vertically (-1),
            # # or horizontal (0).
            yincreasing = cuda.local.array(4, dtype=numba.types.int8)

            # # Array that will hold mask of whether edges are
            # # eligible for intersection tests
            eligible = cuda.local.array(4, dtype=numba.types.int8)

            # # Array that will hold a mask of whether edges
            # # intersect the ray to the right of test point
            intersect = cuda.local.array(4, dtype=numba.types.int8)

            i, j = cuda.grid(2)
            if i < (xs.shape[0] - 1) and j < (ys.shape[0] - 1):
                perform_extend(
                    i, j, plot_height, plot_width, xs, ys,
                    xverts, yverts, yincreasing, eligible, intersect,
                    *aggs_and_cols
                )

        @ngjit
        @self.expand_aggs_and_cols(append)
        def extend_cpu(plot_height, plot_width, xs, ys, *aggs_and_cols):
            # For performance, we initialize all arrays once before the loop

            # xverts/yverts arrays
            xverts = np.zeros(5, dtype=np.int32)
            yverts = np.zeros(5, dtype=np.int32)

            # Array holding whether each edge is increasing
            # vertically (+1), decreasing vertically (-1),
            # or horizontal (0).
            yincreasing = np.zeros(4, dtype=np.int8)

            # Array that will hold mask of whether edges are
            # eligible for intersection tests
            eligible = np.ones(4, dtype=np.int8)

            # Array that will hold a mask of whether edges
            # intersect the ray to the right of test point
            intersect = np.zeros(4, dtype=np.int8)

            y_len, x_len, = xs.shape
            for i in range(x_len - 1):
                for j in range(y_len - 1):
                    perform_extend(
                        i, j, plot_height, plot_width, xs, ys,
                        xverts, yverts, yincreasing, eligible, intersect, *aggs_and_cols
                    )

        def extend(aggs, xr_ds, vt, bounds):
            from datashader.core import LinearAxis
            use_cuda = cupy and isinstance(xr_ds[name].data, cupy.ndarray)

            x_breaks = xr_ds[x_name].values
            y_breaks = xr_ds[y_name].values
            if use_cuda:
                x_breaks = cupy.array(x_breaks)
                y_breaks = cupy.array(y_breaks)

                x_mapper2 = _cuda_mapper(x_mapper)
                y_mapper2 = _cuda_mapper(y_mapper)
            else:
                x_mapper2 = _cuda_mapper(x_mapper)
                y_mapper2 = _cuda_mapper(y_mapper)

            # Convert from bin centers to interval edges
            x_breaks = infer_interval_breaks(x_breaks, axis=1)
            x_breaks = infer_interval_breaks(x_breaks, axis=0)

            y_breaks = infer_interval_breaks(y_breaks, axis=1)
            y_breaks = infer_interval_breaks(y_breaks, axis=0)

            # Scale x and y vertices into integer canvas coordinates
            x0, x1, y0, y1 = bounds
            xspan = x1 - x0
            yspan = y1 - y0

            if x_mapper is LinearAxis.mapper:
                xscaled = (x_breaks - x0) / xspan
            else:
                xscaled = (x_mapper2(x_breaks) - x0) / xspan

            if y_mapper is LinearAxis.mapper:
                yscaled = (y_breaks - y0) / yspan
            else:
                yscaled = (y_mapper2(y_breaks) - y0) / yspan

            plot_height, plot_width = aggs[0].shape[:2]

            xs = (xscaled * plot_width).astype(int)
            ys = (yscaled * plot_height).astype(int)

            coord_dims = xr_ds.coords[x_name].dims
            aggs_and_cols = aggs + info(xr_ds.transpose(*coord_dims))
            if use_cuda:
                do_extend = extend_cuda[cuda_args(xr_ds[name].shape)]
            else:
                do_extend = extend_cpu

            do_extend(
                plot_height, plot_width, xs, ys, *aggs_and_cols
            )

        return extend
