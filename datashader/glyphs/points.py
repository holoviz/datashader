from __future__ import absolute_import, division
import numpy as np
from toolz import memoize

from datashader.glyphs.glyph import Glyph
from datashader.utils import isreal, ngjit

from numba import cuda

try:
    import cudf
    from ..transfer_functions._cuda_utils import cuda_args
except ImportError:
    cudf = None
    cuda_args = None


def values(s):
    if isinstance(s, cudf.Series):
        return s.to_gpu_array(fillna=np.nan)
    else:
        return s.values


class _PointLike(Glyph):
    """Shared methods between Point and Line"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def ndims(self):
        return 1

    @property
    def inputs(self):
        return (self.x, self.y)

    def validate(self, in_dshape):
        if not isreal(in_dshape.measure[str(self.x)]):
            raise ValueError('x must be real')
        elif not isreal(in_dshape.measure[str(self.y)]):
            raise ValueError('y must be real')

    @property
    def x_label(self):
        return self.x

    @property
    def y_label(self):
        return self.y

    def required_columns(self):
        return [self.x, self.y]

    def compute_x_bounds(self, df):
        bounds = self._compute_bounds(df[self.x])
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds = self._compute_bounds(df[self.y])
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin(df[self.x].values).item(),
            np.nanmax(df[self.x].values).item(),
            np.nanmin(df[self.y].values).item(),
            np.nanmax(df[self.y].values).item()]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, 2]), np.nanmax(r[:, 3])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))


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
        @self.expand_aggs_and_cols(append)
        def _perform_extend_points(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols):
            x = xs[i]
            y = ys[i]
            # points outside bounds are dropped; remainder
            # are mapped onto pixels
            if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                xx = int(x_mapper(x) * sx + tx)
                yy = int(y_mapper(y) * sy + ty)
                xi, yi = (xx - 1 if x == xmax else xx,
                          yy - 1 if y == ymax else yy)

                append(i, xi, yi, *aggs_and_cols)

        @ngjit
        @self.expand_aggs_and_cols(append)
        def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols):
            for i in range(xs.shape[0]):
                _perform_extend_points(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols)

        @cuda.jit
        @self.expand_aggs_and_cols(append)
        def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols):
            i = cuda.grid(1)
            if i < xs.shape[0]:
                _perform_extend_points(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols)

        def extend(aggs, df, vt, bounds):
            aggs_and_cols = aggs + info(df)
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds

            if cudf and isinstance(df, cudf.DataFrame):
                xs = df[x_name].to_gpu_array(fillna=np.nan)
                ys = df[y_name].to_gpu_array(fillna=np.nan)
                do_extend = extend_cuda[cuda_args(xs.shape[0])]
            else:
                xs = df[x_name].values
                ys = df[y_name].values
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
            )

        return extend
