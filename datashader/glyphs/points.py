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


class _GeomLike(Glyph):
    def __init__(self, geometry):
        self.geometry = geometry

    @property
    def ndims(self):
        return 1

    @property
    def inputs(self):
        return (self.geometry,)

    @property
    def geom_dtype(self):
        from datashader.geom import GeomDtype
        return GeomDtype

    def validate(self, in_dshape):
        if not isinstance(in_dshape[str(self.geometry)], self.geom_dtype):
            raise ValueError('{col} must be a {typ} array'.format(
                col=self.geometry, typ=self.geom_dtype._type_name
            ))

    @property
    def x_label(self):
        return 'x'

    @property
    def y_label(self):
        return 'y'

    def required_columns(self):
        return [self.geometry]

    def compute_x_bounds(self, df):
        bounds = df[self.geometry].array.bounds_x
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds = df[self.geometry].array.bounds_y
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array(
            [list(df[self.geometry].array.bounds)]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 2])
        y_extents = np.nanmin(r[:, 1]), np.nanmax(r[:, 3])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))


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
            np.nanmin(df[self.x].values),
            np.nanmax(df[self.x].values),
            np.nanmin(df[self.y].values),
            np.nanmax(df[self.y].values)]]
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


class PointGeom(_GeomLike):

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        geometry_name = self.geometry

        @ngjit
        @self.expand_aggs_and_cols(append)
        def _perform_extend_points(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, flat_array, *aggs_and_cols
        ):
            x = flat_array[j]
            y = flat_array[j + 1]
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
        def extend_cpu(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                flat_array, start_indices, *aggs_and_cols
        ):
            n = len(start_indices)
            m = len(flat_array)
            for i in range(n):
                start = start_indices[i]
                stop = start_indices[i + 1] if i < n - 1 else m
                for j in range(start, stop, 2):
                    _perform_extend_points(
                        i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                        flat_array, *aggs_and_cols
                    )

        def extend(aggs, df, vt, bounds):
            aggs_and_cols = aggs + info(df)
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds

            geometry = df[geometry_name].array
            flat_array = geometry.flat_array
            start_indices = geometry.start_indices

            extend_cpu(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                flat_array, start_indices, *aggs_and_cols
            )

        return extend
