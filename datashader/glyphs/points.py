from __future__ import annotations
from packaging.version import Version
import numpy as np
from toolz import memoize

from datashader.glyphs.glyph import Glyph
from datashader.utils import isreal, ngjit

from numba import cuda

try:
    import cudf
    from ..transfer_functions._cuda_utils import cuda_args
except Exception:
    cudf = None
    cuda_args = None

try:
    from geopandas.array import GeometryDtype as gpd_GeometryDtype
except ImportError:
    gpd_GeometryDtype = type(None)

try:
    import spatialpandas
except Exception:
    spatialpandas = None


def values(s):
    if isinstance(s, cudf.Series):
        if Version(cudf.__version__) >= Version("22.02"):
            return s.to_cupy(na_value=np.nan)
        else:
            return s.to_gpu_array(fillna=np.nan)

    else:
        return s.values


class _GeometryLike(Glyph):
    def __init__(self, geometry):
        self.geometry = geometry
        self._cached_bounds = None

    @property
    def ndims(self):
        return 1

    @property
    def inputs(self):
        return (self.geometry,)

    @property
    def geom_dtypes(self):
        if spatialpandas:
            from spatialpandas.geometry import GeometryDtype
            return (GeometryDtype,)
        else:
            return ()  # Empty tuple

    def validate(self, in_dshape):
        if not isinstance(in_dshape[str(self.geometry)], self.geom_dtypes):
            raise ValueError(
                '{col} must be an array with one of the following types: {typs}'.format(
                    col=self.geometry,
                    typs=', '.join(typ.__name__ for typ in self.geom_dtypes)
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
        col = df[self.geometry]
        if isinstance(col.dtype, gpd_GeometryDtype):
            # geopandas
            if self._cached_bounds is None:
                self._cached_bounds = col.total_bounds
            bounds = self._cached_bounds[::2]
        else:
            # spatialpandas
            bounds = col.array.total_bounds_x
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        col = df[self.geometry]
        if isinstance(col.dtype, gpd_GeometryDtype):
            # geopandas
            if self._cached_bounds is None:
                self._cached_bounds = col.total_bounds
            bounds = self._cached_bounds[1::2]
        else:
            # spatialpandas
            bounds = col.array.total_bounds_y
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):
        total_bounds = ddf[self.geometry].total_bounds
        x_extents = (total_bounds[0], total_bounds[2])
        y_extents = (total_bounds[1], total_bounds[3])

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
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2,
                      _antialias_stage_2_funcs):
        x_name = self.x
        y_name = self.y

        @ngjit
        @self.expand_aggs_and_cols(append)
        def _perform_extend_points(i, sx, tx, sy, ty, xmin, xmax,
                                   ymin, ymax, xs, ys, xxmax, yymax,
                                   *aggs_and_cols):
            x = xs[i]
            y = ys[i]

            # points outside bounds are dropped; remainder
            # are mapped onto pixels
            if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                xx = int(x_mapper(x) * sx + tx)
                yy = int(y_mapper(y) * sy + ty)

                xi, yi = (xxmax-1 if xx >= xxmax else xx,
                          yymax-1 if yy >= yymax else yy)
                append(i, xi, yi, *aggs_and_cols)

        @ngjit
        @self.expand_aggs_and_cols(append)
        def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                       xxmax, yymax, *aggs_and_cols):
            for i in range(xs.shape[0]):
                _perform_extend_points(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                       xs, ys, xxmax, yymax, *aggs_and_cols)

        @cuda.jit
        @self.expand_aggs_and_cols(append)
        def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                        xxmax, yymax, *aggs_and_cols):
            i = cuda.grid(1)
            if i < xs.shape[0]:
                _perform_extend_points(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                       xs, ys, xxmax, yymax, *aggs_and_cols)

        def extend(aggs, df, vt, bounds):
            yymax, xxmax = aggs[0].shape[:2]
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds

            if cudf and isinstance(df, cudf.DataFrame):
                xs = values(df[x_name])
                ys = values(df[y_name])
                do_extend = extend_cuda[cuda_args(xs.shape[0])]
            else:
                xs = df[x_name].values
                ys = df[y_name].values
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, xxmax, yymax, *aggs_and_cols
            )

        return extend


class MultiPointGeoPandas(_GeometryLike):
    # geopandas must be available if a GeoPandasPointGeometry object is created.
    @property
    def geom_dtypes(self):
        from geopandas.array import GeometryDtype
        return (GeometryDtype,)

    @memoize
    def _build_extend(
        self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs,
    ):
        # Lazy import shapely. Cannot get here if geopandas and shapely are not available.
        import shapely

        geometry_name = self.geometry

        @ngjit
        @self.expand_aggs_and_cols(append)
        def _perform_extend_points(
            i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols
        ):
            x = values[j]
            y = values[j + 1]
            # points outside bounds are dropped; remainder
            # are mapped onto pixels
            if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                xx = int(x_mapper(x) * sx + tx)
                yy = int(y_mapper(y) * sy + ty)
                xi, yi = (xx - 1 if x == xmax else xx,
                          yy - 1 if y == ymax else yy)

                append(i, xi, yi, *aggs_and_cols)

        def extend(aggs, df, vt, bounds):
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            geometry = df[geometry_name].array

            ragged = shapely.to_ragged_array(geometry)
            geometry_type = ragged[0]

            if geometry_type not in (shapely.GeometryType.MULTIPOINT, shapely.GeometryType.POINT):
                raise ValueError(
                    "Canvas.points supports GeoPandas geometry types of POINT and MULTIPOINT, "
                    f"not {repr(geometry_type)}")

            coords = ragged[1].ravel()  # No offsets required if POINT not MULTIPOINT
            if geometry_type == shapely.GeometryType.POINT:
                extend_point_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, coords, *aggs_and_cols)
            else:
                offsets = ragged[2][0]
                extend_multipoint_cpu(
                    sx, tx, sy, ty, xmin, xmax, ymin, ymax, coords, offsets, *aggs_and_cols)

        @ngjit
        @self.expand_aggs_and_cols(append)
        def extend_multipoint_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, offsets, *aggs_and_cols,
        ):
            for i in range(len(offsets) - 1):
                start = offsets[i]
                stop = offsets[i+1]
                for j in range(start, stop):
                    _perform_extend_points(
                        i, 2*j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols,
                    )

        @ngjit
        @self.expand_aggs_and_cols(append)
        def extend_point_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols):
            n = len(values) // 2
            for i in range(n):
                _perform_extend_points(
                    i, 2*i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols,
                )

        return extend


class MultiPointGeometry(_GeometryLike):
    # spatialpandas must be available if a MultiPointGeometry object is created.

    @property
    def geom_dtypes(self):
        from spatialpandas.geometry import PointDtype, MultiPointDtype
        return PointDtype, MultiPointDtype

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2,
                      _antialias_stage_2_funcs):
        geometry_name = self.geometry

        @ngjit
        @self.expand_aggs_and_cols(append)
        def _perform_extend_points(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols
        ):
            x = values[j]
            y = values[j + 1]
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
        def extend_point_cpu(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                values, missing, eligible_inds, *aggs_and_cols
        ):
            for i in eligible_inds:
                if missing[i] is True:
                    continue
                _perform_extend_points(
                    i, 2 * i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    values, *aggs_and_cols
                )

        @ngjit
        @self.expand_aggs_and_cols(append)
        def extend_multipoint_cpu(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                values, missing, offsets, eligible_inds, *aggs_and_cols
        ):
            for i in eligible_inds:
                if missing[i] is True:
                    continue
                start = offsets[i]
                stop = offsets[i + 1]
                for j in range(start, stop, 2):
                    _perform_extend_points(
                        i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                        values, *aggs_and_cols
                    )

        def extend(aggs, df, vt, bounds):
            from spatialpandas.geometry import PointArray

            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds

            geometry = df[geometry_name].array

            if geometry._sindex is not None:
                # Compute indices of potentially intersecting polygons using
                # geometry's R-tree if there is one
                eligible_inds = geometry.sindex.intersects((xmin, ymin, xmax, ymax))
            else:
                # Otherwise, process all indices
                eligible_inds = np.arange(0, len(geometry), dtype='uint32')

            missing = geometry.isna()

            if isinstance(geometry, PointArray):
                values = geometry.flat_values
                extend_point_cpu(
                    sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    values, missing, eligible_inds, *aggs_and_cols
                )
            else:
                values = geometry.buffer_values
                offsets = geometry.buffer_offsets[0]

                extend_multipoint_cpu(
                    sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    values, missing, offsets, eligible_inds, *aggs_and_cols
                )

        return extend
