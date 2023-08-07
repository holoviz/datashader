from __future__ import annotations
import math
import numpy as np
from toolz import memoize

from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types


try:
    import cudf
    import cupy as cp
    from ..transfer_functions._cuda_utils import cuda_args
except ImportError:
    cudf = None
    cuda_args = None

try:
    import spatialpandas
except Exception:
    spatialpandas = None


class _AntiAliasedLine(object):
    """ Methods common to all lines. """
    _line_width = 0  # Use antialiasing if > 0.

    def set_line_width(self, line_width):
        self._line_width = line_width
        if hasattr(self, "antialiased"):
            self.antialiased = (line_width > 0)

    def _build_extend(self, x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs):
        return self._internal_build_extend(
                x_mapper, y_mapper, info, append, self._line_width, antialias_stage_2, antialias_stage_2_funcs)


class LineAxis0(_PointLike, _AntiAliasedLine):
    """A line, with vertices defined by ``x`` and ``y``.

    Parameters
    ----------
    x, y : str
        Column names for the x and y coordinates of each vertex.
    """
    @memoize
    def _internal_build_extend(
            self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        antialias = line_width > 0
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(
            x_mapper, y_mapper, antialias)
        overwrite, use_2_stage_agg = two_stage_agg(antialias_stage_2)
        if not use_2_stage_agg:
            antialias_stage_2_funcs = None
        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols, line_width, overwrite
        )
        extend_cpu, extend_cuda = _build_extend_line_axis0(
            draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs,
        )
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, x_name)
                ys = self.to_cupy_array(df, y_name)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, x_name].to_numpy()
                ys = df.loc[:, y_name].to_numpy()
                do_extend = extend_cpu

            # line may be clipped, then mapped to pixels
            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                xs, ys, plot_start, antialias_stage_2, *aggs_and_cols
            )

        return extend


class LineAxis0Multi(_PointLike, _AntiAliasedLine):
    """
    """

    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(xcol)]) for xcol in self.x]):
            raise ValueError('x columns must be real')
        elif not all([isreal(in_dshape.measure[str(ycol)]) for ycol in self.y]):
            raise ValueError('y columns must be real')

        if len(self.x) != len(self.y):
            raise ValueError(
                f'x and y coordinate lengths do not match: {len(self.x)} != {len(self.y)}')

    @property
    def x_label(self):
        return 'x'

    @property
    def y_label(self):
        return 'y'

    def required_columns(self):
        return self.x + self.y

    def compute_x_bounds(self, df):
        bounds_list = [self._compute_bounds(df[x])
                       for x in self.x]
        mins, maxes = zip(*bounds_list)
        return self.maybe_expand_bounds((min(mins), max(maxes)))

    def compute_y_bounds(self, df):
        bounds_list = [self._compute_bounds(df[y])
                       for y in self.y]
        mins, maxes = zip(*bounds_list)
        return self.maybe_expand_bounds((min(mins), max(maxes)))

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin([np.nanmin(df[c].values).item() for c in self.x]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.x]),
            np.nanmin([np.nanmin(df[c].values).item() for c in self.y]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.y])]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, 2]), np.nanmax(r[:, 3])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _internal_build_extend(
            self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        antialias = line_width > 0
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(
            x_mapper, y_mapper, antialias)
        overwrite, use_2_stage_agg = two_stage_agg(antialias_stage_2)
        if not use_2_stage_agg:
            antialias_stage_2_funcs = None
        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols, line_width, overwrite
        )
        extend_cpu, extend_cuda = _build_extend_line_axis0_multi(
            draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs,
        )

        x_names = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, x_names)
                ys = self.to_cupy_array(df, y_names)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys = df.loc[:, list(y_names)].to_numpy()
                do_extend = extend_cpu

            # line may be clipped, then mapped to pixels
            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                xs, ys, plot_start, antialias_stage_2, *aggs_and_cols,
            )

        return extend


class LinesAxis1(_PointLike, _AntiAliasedLine):
    """A collection of lines (on line per row) with vertices defined
    by the lists of columns in ``x`` and ``y``

    Parameters
    ----------
    x, y : list
        Lists of column names for the x and y coordinates
    """

    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(xcol)])
                    for xcol in self.x]):
            raise ValueError('x columns must be real')
        elif not all([isreal(in_dshape.measure[str(ycol)])
                      for ycol in self.y]):
            raise ValueError('y columns must be real')

        unique_x_measures = set(in_dshape.measure[str(xcol)]
                                for xcol in self.x)
        if len(unique_x_measures) > 1:
            raise ValueError('x columns must have the same data type')

        unique_y_measures = set(in_dshape.measure[str(ycol)]
                                for ycol in self.y)
        if len(unique_y_measures) > 1:
            raise ValueError('y columns must have the same data type')

        if len(self.x) != len(self.y):
            raise ValueError(
                f'x and y coordinate lengths do not match: {len(self.x)} != {len(self.y)}')

    def required_columns(self):
        return self.x + self.y

    @property
    def x_label(self):
        return 'x'

    @property
    def y_label(self):
        return 'y'

    def compute_x_bounds(self, df):
        xs = tuple(df[xlabel] for xlabel in self.x)

        bounds_list = [self._compute_bounds(xcol) for xcol in xs]
        mins, maxes = zip(*bounds_list)

        return self.maybe_expand_bounds((min(mins), max(maxes)))

    def compute_y_bounds(self, df):
        ys = tuple(df[ylabel] for ylabel in self.y)

        bounds_list = [self._compute_bounds(ycol) for ycol in ys]
        mins, maxes = zip(*bounds_list)

        return self.maybe_expand_bounds((min(mins), max(maxes)))

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin([np.nanmin(df[c].values).item() for c in self.x]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.x]),
            np.nanmin([np.nanmin(df[c].values).item() for c in self.y]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.y])]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, 2]), np.nanmax(r[:, 3])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _internal_build_extend(
            self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        antialias = line_width > 0
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(
            x_mapper, y_mapper, antialias)
        overwrite, use_2_stage_agg = two_stage_agg(antialias_stage_2)
        if not use_2_stage_agg:
            antialias_stage_2_funcs = None
        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols, line_width, overwrite
        )
        extend_cpu, extend_cuda = _build_extend_line_axis1_none_constant(
            draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs,
        )
        x_names = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, x_names)
                ys = self.to_cupy_array(df, y_names)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys = df.loc[:, list(y_names)].to_numpy()
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols
            )

        return extend


class LinesAxis1XConstant(LinesAxis1):
    """
    """
    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(ycol)]) for ycol in self.y]):
            raise ValueError('y columns must be real')

        unique_y_measures = set(in_dshape.measure[str(ycol)]
                                for ycol in self.y)
        if len(unique_y_measures) > 1:
            raise ValueError('y columns must have the same data type')

        if len(self.x) != len(self.y):
            raise ValueError(
                f'x and y coordinate lengths do not match: {len(self.x)} != {len(self.y)}')

    def required_columns(self):
        return self.y

    def compute_x_bounds(self, *args):
        x_min = np.nanmin(self.x)
        x_max = np.nanmax(self.x)
        return self.maybe_expand_bounds((x_min, x_max))

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin([np.nanmin(df[c].values).item() for c in self.y]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.y])]]
        )).compute()

        y_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])

        return (self.compute_x_bounds(),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _internal_build_extend(
            self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        antialias = line_width > 0
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(
            x_mapper, y_mapper, antialias)
        overwrite, use_2_stage_agg = two_stage_agg(antialias_stage_2)
        if not use_2_stage_agg:
            antialias_stage_2_funcs = None
        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols, line_width, overwrite
        )
        extend_cpu, extend_cuda = _build_extend_line_axis1_x_constant(
            draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs,
        )

        x_values = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                xs = cp.asarray(x_values)
                ys = self.to_cupy_array(df, y_names)
                do_extend = extend_cuda[cuda_args(ys.shape)]
            else:
                xs = x_values
                ys = df.loc[:, list(y_names)].to_numpy()
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols
            )

        return extend


class LinesAxis1YConstant(LinesAxis1):
    """
    """
    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(xcol)]) for xcol in self.x]):
            raise ValueError('x columns must be real')

        unique_x_measures = set(in_dshape.measure[str(xcol)]
                                for xcol in self.x)
        if len(unique_x_measures) > 1:
            raise ValueError('x columns must have the same data type')

        if len(self.x) != len(self.y):
            raise ValueError(
                f'x and y coordinate lengths do not match: {len(self.x)} != {len(self.y)}')

    def required_columns(self):
        return self.x

    def compute_y_bounds(self, *args):
        y_min = np.nanmin(self.y)
        y_max = np.nanmax(self.y)
        return self.maybe_expand_bounds((y_min, y_max))

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin([np.nanmin(df[c].values).item() for c in self.x]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.x])]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])

        return (self.maybe_expand_bounds(x_extents),
                self.compute_y_bounds())

    @memoize
    def _internal_build_extend(
            self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        antialias = line_width > 0
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(
            x_mapper, y_mapper, antialias)
        overwrite, use_2_stage_agg = two_stage_agg(antialias_stage_2)
        if not use_2_stage_agg:
            antialias_stage_2_funcs = None
        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols, line_width, overwrite
        )
        extend_cpu, extend_cuda = _build_extend_line_axis1_y_constant(
            draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs,
        )

        x_names = self.x
        y_values = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, x_names)
                ys = cp.asarray(y_values)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys = y_values
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols
            )

        return extend


class LinesAxis1Ragged(_PointLike, _AntiAliasedLine):
    def validate(self, in_dshape):
        try:
            from datashader.datatypes import RaggedDtype
        except ImportError:
            RaggedDtype = type(None)

        if not isinstance(in_dshape[str(self.x)], RaggedDtype):
            raise ValueError('x must be a RaggedArray')
        elif not isinstance(in_dshape[str(self.y)], RaggedDtype):
            raise ValueError('y must be a RaggedArray')

    def required_columns(self):
        return (self.x,) + (self.y,)

    def compute_x_bounds(self, df):
        bounds = self._compute_bounds(df[self.x].array.flat_array)
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds = self._compute_bounds(df[self.y].array.flat_array)
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin(df[self.x].array.flat_array).item(),
            np.nanmax(df[self.x].array.flat_array).item(),
            np.nanmin(df[self.y].array.flat_array).item(),
            np.nanmax(df[self.y].array.flat_array).item()]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, 2]), np.nanmax(r[:, 3])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _internal_build_extend(
            self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        antialias = line_width > 0
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(
            x_mapper, y_mapper, antialias)
        overwrite, use_2_stage_agg = two_stage_agg(antialias_stage_2)
        if not use_2_stage_agg:
            antialias_stage_2_funcs = None
        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols, line_width, overwrite
        )
        extend_cpu = _build_extend_line_axis1_ragged(
            draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs,
        )
        x_name = self.x
        y_name = self.y
        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds

            xs = df[x_name].array
            ys = df[y_name].array

            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            # line may be clipped, then mapped to pixels
            extend_cpu(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols
            )

        return extend


class LineAxis1Geometry(_GeometryLike, _AntiAliasedLine):
    # spatialpandas must be available if a LineAxis1Geometry object is created.

    @property
    def geom_dtypes(self):
        from spatialpandas.geometry import (
            LineDtype, MultiLineDtype, RingDtype, PolygonDtype,
            MultiPolygonDtype
        )
        return (LineDtype, MultiLineDtype, RingDtype,
                PolygonDtype, MultiPolygonDtype)

    @memoize
    def _internal_build_extend(
            self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        from spatialpandas.geometry import (
            PolygonArray, MultiPolygonArray, RingArray
        )
        antialias = line_width > 0
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(
            x_mapper, y_mapper, antialias)
        overwrite, use_2_stage_agg = two_stage_agg(antialias_stage_2)
        if not use_2_stage_agg:
            antialias_stage_2_funcs = None
        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols, line_width, overwrite
        )
        perform_extend_cpu = _build_extend_line_axis1_geometry(
            draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs,
        )
        geometry_name = self.geometry

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            geom_array = df[geometry_name].array

            # Use type to decide whether geometry represents a closed .
            # We skip for closed geometries so as not to double count the first/last
            # pixel
            if isinstance(geom_array, (PolygonArray, MultiPolygonArray)):
                # Convert polygon array to multi line of boundary
                geom_array = geom_array.boundary
                closed_rings = True
            elif isinstance(geom_array, RingArray):
                closed_rings = True
            else:
                closed_rings = False

            perform_extend_cpu(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                geom_array, closed_rings, antialias_stage_2, *aggs_and_cols
            )

        return extend


def _build_map_onto_pixel_for_line(x_mapper, y_mapper, want_antialias=False):
    @ngjit
    def map_onto_pixel_snap(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x, y):
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
        xx = int(x_mapper(x) * sx + tx)
        yy = int(y_mapper(y) * sy + ty)

        # Note that sx and tx were designed so that
        # x_mapper(xmax) * sx + tx equals the width of the canvas in pixels
        #
        # Likewise, sy and ty were designed so that
        # y_mapper(ymax) * sy + ty equals the height of the canvas in pixels
        #
        # We round these results to integers (rather than casting to integers
        # with the int constructor) to handle cases where floating-point
        # precision errors results in a value just under the integer number
        # of pixels.
        xxmax = round(x_mapper(xmax) * sx + tx)
        yymax = round(y_mapper(ymax) * sy + ty)

        return (xx - 1 if xx == xxmax else xx,
                yy - 1 if yy == yymax else yy)

    @ngjit
    def map_onto_pixel_no_snap(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x, y):
        xx = x_mapper(x)*sx + tx - 0.5
        yy = y_mapper(y)*sy + ty - 0.5
        return xx, yy

    if want_antialias:
        return map_onto_pixel_no_snap
    else:
        return map_onto_pixel_snap


@ngjit
def _liang_barsky(xmin, xmax, ymin, ymax, x0, x1, y0, y1, skip):
    """ An implementation of the Liang-Barsky line clipping algorithm.

    https://en.wikipedia.org/wiki/Liang%E2%80%93Barsky_algorithm

    """
    # Check if line is fully outside viewport
    if x0 < xmin and x1 < xmin:
        skip = True
    elif x0 > xmax and x1 > xmax:
        skip = True
    elif y0 < ymin and y1 < ymin:
        skip = True
    elif y0 > ymax and y1 > ymax:
        skip = True

    t0, t1 = 0, 1
    dx1 = x1 - x0
    t0, t1, accept = _clipt(-dx1, x0 - xmin, t0, t1)
    if not accept:
        skip = True
    t0, t1, accept = _clipt(dx1, xmax - x0, t0, t1)
    if not accept:
        skip = True
    dy1 = y1 - y0
    t0, t1, accept = _clipt(-dy1, y0 - ymin, t0, t1)
    if not accept:
        skip = True
    t0, t1, accept = _clipt(dy1, ymax - y0, t0, t1)
    if not accept:
        skip = True
    if t1 < 1:
        clipped_end = True
        x1 = x0 + t1 * dx1
        y1 = y0 + t1 * dy1
    else:
        clipped_end = False
    if t0 > 0:
        # If x0 is clipped, we need to plot the new start
        clipped_start = True
        x0 = x0 + t0 * dx1
        y0 = y0 + t0 * dy1
    else:
        clipped_start = False

    return x0, x1, y0, y1, skip, clipped_start, clipped_end


@ngjit
def _clipt(p, q, t0, t1):
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
def _clamp(x, low, high):
    # Clamp ``x`` in the range ``low`` to ``high``.
    return max(low, min(x, high))


@ngjit
def _linearstep(edge0, edge1, x):
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t


@ngjit
def _x_intercept(y, cx0, cy0, cx1, cy1):
    # Return x value of intercept between line at constant y and line
    # between corner points.
    if cy0 == cy1:
        # Line is horizontal, return the "upper", i.e. right-hand, end of it.
        return cx1
    frac = (y - cy0) / (cy1 - cy0)  # In range 0..1
    return cx0 + frac*(cx1 - cx0)


def _build_full_antialias(expand_aggs_and_cols):
    """Specialize antialiased line drawing algorithm for a given append/axis combination"""
    @ngjit
    @expand_aggs_and_cols
    def _full_antialias(line_width, overwrite, i, x0, x1, y0, y1,
                        segment_start, segment_end, xm, ym, append,
                        nx, ny, buffer, *aggs_and_cols):
        """Draw an antialiased line segment.

        If overwrite=True can overwrite each pixel multiple times because
        using max for the overwriting.  If False can only write each pixel
        once per segment and its previous segment.
        Argument xm, ym are only valid if overwrite and segment_start are False.
        """
        if x0 == x1 and y0 == y1:
            return

        # Scan occurs in y-direction. But wish to scan in the shortest direction,
        # so if |x0-x1| < |y0-y1| then flip (x,y) coords for maths and flip back
        # again before setting pixels.
        flip_xy = abs(x0-x1) < abs(y0-y1)
        if flip_xy:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            xm, ym = ym, xm

        scale = 1.0

        # line_width less than 1 is rendered as 1 but with lower intensity.
        if line_width < 1.0:
            scale *= line_width
            line_width = 1.0

        aa = 1.0
        halfwidth = 0.5*(line_width + aa)

        # Want y0 <= y1, so switch vertical direction if this is not so.
        flip_order = y1 < y0 or (y1 == y0 and x1 < x0)

        # Start (x0, y0), end (y0, y1)
        #       c1 +-------------+ c2          along    | right
        # (x0, y0) | o         o | (x1, y1)    vector   | vector
        #       c0 +-------------+ c3          ---->    v

        alongx = float(x1 - x0)
        alongy = float(y1 - y0)  # Always +ve
        length = math.sqrt(alongx**2 + alongy**2)
        alongx /= length
        alongy /= length

        rightx = alongy
        righty = -alongx

        # 4 corners, x and y.  Uses buffer, which must have length 8.  Order of coords is
        # (x0, x1, x2, x3, y0, y1, y2, y3).  Each CPU/GPU thread has its own local buffer
        # so there is no cross-talk.  Contents of buffer are written and read within the
        # lifetime of this function, so it doesn't matter what they are before this
        # function is called or after it returns.
        if flip_order:
            buffer[0] = x1 - halfwidth*( rightx - alongx)
            buffer[1] = x1 - halfwidth*(-rightx - alongx)
            buffer[2] = x0 - halfwidth*(-rightx + alongx)
            buffer[3] = x0 - halfwidth*( rightx + alongx)
            buffer[4] = y1 - halfwidth*( righty - alongy)
            buffer[5] = y1 - halfwidth*(-righty - alongy)
            buffer[6] = y0 - halfwidth*(-righty + alongy)
            buffer[7] = y0 - halfwidth*( righty + alongy)
        else:
            buffer[0] = x0 + halfwidth*( rightx - alongx)
            buffer[1] = x0 + halfwidth*(-rightx - alongx)
            buffer[2] = x1 + halfwidth*(-rightx + alongx)
            buffer[3] = x1 + halfwidth*( rightx + alongx)
            buffer[4] = y0 + halfwidth*( righty - alongy)
            buffer[5] = y0 + halfwidth*(-righty - alongy)
            buffer[6] = y1 + halfwidth*(-righty + alongy)
            buffer[7] = y1 + halfwidth*( righty + alongy)

        xmax = nx-1
        ymax = ny-1
        if flip_xy:
            xmax, ymax = ymax, xmax

        # Index of lowest-y point.
        if flip_order:
            lowindex = 0 if x0 > x1 else 1
        else:
            lowindex = 0 if x1 > x0 else 1

        if not overwrite and not segment_start:
            prev_alongx = x0 - xm
            prev_alongy = y0 - ym
            prev_length = math.sqrt(prev_alongx**2 + prev_alongy**2)
            if prev_length > 0.0:
                prev_alongx /= prev_length
                prev_alongy /= prev_length
                prev_rightx = prev_alongy
                prev_righty = -prev_alongx
            else:
                overwrite = True

        # y limits of scan.
        ystart = _clamp(math.ceil(buffer[4 + lowindex]), 0, ymax)
        yend = _clamp(math.floor(buffer[4 + (lowindex+2) % 4]), 0, ymax)
        # Need to know which edges are to left and right; both will change.
        ll = lowindex  # Index of lower point of left edge.
        lu = (ll + 1) % 4  # Index of upper point of left edge.
        rl = lowindex  # Index of lower point of right edge.
        ru = (rl + 3) % 4  # Index of upper point of right edge.
        for y in range(ystart, yend+1):
            if ll == lowindex and y > buffer[4 + lu]:
                ll = lu
                lu = (ll + 1) % 4
            if rl == lowindex and y > buffer[4 + ru]:
                rl = ru
                ru = (rl + 3) % 4
            # Find x limits of scan at this y.
            xleft = _clamp(math.ceil(_x_intercept(
                y, buffer[ll], buffer[4+ll], buffer[lu], buffer[4+lu])), 0, xmax)
            xright = _clamp(math.floor(_x_intercept(
                y, buffer[rl], buffer[4+rl], buffer[ru], buffer[4+ru])), 0, xmax)
            for x in range(xleft, xright+1):
                along = (x-x0)*alongx + (y-y0)*alongy  # dot product
                prev_correction = False
                if along < 0.0:
                    # Before start of segment
                    if overwrite or segment_start or (x-x0)*prev_alongx + (y-y0)*prev_alongy > 0.0:
                        distance = math.sqrt((x-x0)**2 + (y-y0)**2)  # round join/end cap
                    else:
                        continue
                elif along > length:
                    # After end of segment
                    if overwrite or segment_end:
                        distance = math.sqrt((x-x1)**2 + (y-y1)**2)  # round join/end cap
                    else:
                        continue
                else:
                    # Within segment
                    distance = abs((x-x0)*rightx + (y-y0)*righty)
                    if not overwrite and not segment_start and \
                            -prev_length <= (x-x0)*prev_alongx + (y-y0)*prev_alongy <= 0.0 and \
                            abs((x-x0)*prev_rightx + (y-y0)*prev_righty) <= halfwidth:
                        prev_correction = True
                value = 1.0 - _linearstep(0.5*(line_width - aa), halfwidth, distance)
                value *= scale
                if prev_correction:
                    # Already set pixel from previous segment, need to correct it
                    prev_distance = abs((x-x0)*prev_rightx + (y-y0)*prev_righty)
                    prev_value = 1.0 - _linearstep(0.5*(line_width - aa), halfwidth, prev_distance)
                    prev_value *= scale
                    value = value - prev_value  # Correction from previous segment.
                if value > 0.0:
                    xx, yy = (y, x) if flip_xy else (x, y)
                    append(i, xx, yy, value, *aggs_and_cols)

    return _full_antialias


def _build_bresenham(expand_aggs_and_cols):
    """Specialize a bresenham kernel for a given append/axis combination"""
    @ngjit
    @expand_aggs_and_cols
    def _bresenham(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start,
                   x0, x1, y0, y1, clipped, append, *aggs_and_cols):
        """Draw a line segment using Bresenham's algorithm
        This method plots a line segment with integer coordinates onto a pixel
        grid.
        """
        dx = x1 - x0
        ix = (dx > 0) - (dx < 0)
        dx = abs(dx) * 2

        dy = y1 - y0
        iy = (dy > 0) - (dy < 0)
        dy = abs(dy) * 2

        # If vertices weren't clipped and are concurrent in integer space,
        # call append and return, so that the second vertex won't be hit below.
        if not clipped and not (dx | dy):
            append(i, x0, y0, *aggs_and_cols)
            return

        if segment_start:
            append(i, x0, y0, *aggs_and_cols)

        if dx >= dy:
            error = 2 * dy - dx
            while x0 != x1:
                if error >= 0 and (error or ix > 0):
                    error -= 2 * dx
                    y0 += iy
                error += 2 * dy
                x0 += ix
                append(i, x0, y0, *aggs_and_cols)
        else:
            error = 2 * dx - dy
            while y0 != y1:
                if error >= 0 and (error or iy > 0):
                    error -= 2 * dy
                    x0 += ix
                error += 2 * dx
                y0 += iy
                append(i, x0, y0, *aggs_and_cols)
    return _bresenham

def _build_draw_segment(append, map_onto_pixel, expand_aggs_and_cols, line_width, overwrite):
    """Specialize a line plotting kernel for a given append/axis combination"""

    if line_width > 0.0:
        _bresenham = None
        _full_antialias = _build_full_antialias(expand_aggs_and_cols)
    else:
        _bresenham = _build_bresenham(expand_aggs_and_cols)
        _full_antialias = None

    @ngjit
    @expand_aggs_and_cols
    def draw_segment(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, segment_end,
            x0, x1, y0, y1, xm, ym, buffer, *aggs_and_cols
    ):
        # xm, ym are only valid if segment_start is True.

        # buffer is a length-8 float64 array if antialiasing is to be used,
        # or None otherwise. It is allocated in the appropriate extend_cpu or
        # extend_cuda function so that it is of the correct type (numpy or
        # cupy) and that there is one per CPU/GPU thread.

        # NOTE: The slightly bizarre variable versioning herein for variables
        # x0, y0, y0, y1 is to deal with Numba not having SSA form prior to
        # version 0.49.0. The result of lack of SSA is that the type inference
        # algorithms would widen types that are multiply defined as would be the
        # case in code such as `x, y = function(x, y)` if the function returned
        # a wider type for x, y then the input x, y.
        skip = False

        # If any of the coordinates are NaN, there's a discontinuity.
        # Skip the entire segment.
        if isnull(x0) or isnull(y0) or isnull(x1) or isnull(y1):
            skip = True
        # Use Liang-Barsky to clip the segment to a bounding box
        x0_1, x1_1, y0_1, y1_1, skip, clipped_start, clipped_end = \
            _liang_barsky(xmin, xmax, ymin, ymax, x0, x1, y0, y1, skip)

        if not skip:
            clipped = clipped_start or clipped_end
            segment_start = segment_start or clipped_start
            x0_2, y0_2 = map_onto_pixel(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0_1, y0_1
            )
            x1_2, y1_2 = map_onto_pixel(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax, x1_1, y1_1
            )
            if line_width > 0.0:
                if segment_start:
                    xm_2 = ym_2 = 0.0
                else:
                    xm_2, ym_2 = map_onto_pixel(
                        sx, tx, sy, ty, xmin, xmax, ymin, ymax, xm, ym)
                nx = round((xmax - xmin)*sx)
                ny = round((ymax - ymin)*sy)
                _full_antialias(line_width, overwrite, i, x0_2, x1_2, y0_2, y1_2,
                                segment_start, segment_end, xm_2, ym_2, append,
                                nx, ny, buffer, *aggs_and_cols)
            else:
                _bresenham(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                           segment_start, x0_2, x1_2, y0_2, y1_2,
                           clipped, append, *aggs_and_cols)

    return draw_segment

def _build_extend_line_axis0(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs):
    use_2_stage_agg = antialias_stage_2_funcs is not None

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_line(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                            plot_start, xs, ys, buffer, *aggs_and_cols):
        x0 = xs[i]
        y0 = ys[i]
        x1 = xs[i + 1]
        y1 = ys[i + 1]
        segment_start = (plot_start if i == 0 else
                         (isnull(xs[i - 1]) or isnull(ys[i - 1])))

        segment_end = (i == len(xs)-2) or isnull(xs[i+2]) or isnull(ys[i+2])

        if segment_start or use_2_stage_agg:
            xm = 0.0
            ym = 0.0
        else:
            xm = xs[i-1]
            ym = ys[i-1]

        draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                     segment_start, segment_end, x0, x1, y0, y1,
                     xm, ym, buffer, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                   xs, ys, plot_start, antialias_stage_2, *aggs_and_cols):
        """Aggregate along a line formed by ``xs`` and ``ys``"""
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        nrows = xs.shape[0]
        for i in range(nrows - 1):
            perform_extend_line(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                plot_start, xs, ys, buffer, *aggs_and_cols)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys, plot_start, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = cuda.local.array(8, nb_types.float64) if antialias else None
        i = cuda.grid(1)
        if i < xs.shape[0] - 1:
            perform_extend_line(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                plot_start, xs, ys, buffer, *aggs_and_cols)

    return extend_cpu, extend_cuda


def _build_extend_line_axis0_multi(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs):
    if antialias_stage_2_funcs is not None:
        aa_stage_2_accumulate, aa_stage_2_clear, aa_stage_2_copy_back = antialias_stage_2_funcs
    use_2_stage_agg = antialias_stage_2_funcs is not None

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                            plot_start, xs, ys, buffer, *aggs_and_cols):
        x0 = xs[i, j]
        y0 = ys[i, j]
        x1 = xs[i + 1, j]
        y1 = ys[i + 1, j]
        segment_start = (plot_start if i == 0 else
                         (isnull(xs[i - 1, j]) or isnull(ys[i - 1, j])))

        segment_end = (i == len(xs)-2) or isnull(xs[i+2, j]) or isnull(ys[i+2, j])

        if segment_start or use_2_stage_agg:
            xm = 0.0
            ym = 0.0
        else:
            xm = xs[i-1, j]
            ym = ys[i-1, j]

        draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                     segment_start, segment_end, x0, x1, y0, y1,
                     xm, ym, buffer, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                   plot_start, antialias_stage_2, *aggs_and_cols):
        """Aggregate along a line formed by ``xs`` and ``ys``"""
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        nrows, ncols = xs.shape

        for j in range(ncols):
            for i in range(nrows - 1):
                perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                    plot_start, xs, ys, buffer, *aggs_and_cols)

    def extend_cpu_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                                  plot_start, antialias_stage_2, *aggs_and_cols):
        """Aggregate along a line formed by ``xs`` and ``ys``"""
        n_aggs = len(antialias_stage_2[0])
        aggs_and_accums = tuple((agg, agg.copy()) for agg in aggs_and_cols[:n_aggs])
        cpu_antialias_2agg_impl(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                                plot_start, antialias_stage_2, aggs_and_accums, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def cpu_antialias_2agg_impl(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                                plot_start, antialias_stage_2, aggs_and_accums, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None

        nrows, ncols = xs.shape
        for j in range(ncols):
            for i in range(nrows - 1):
                perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                    plot_start, xs, ys, buffer, *aggs_and_cols)

            if ncols == 1:
                return

            aa_stage_2_accumulate(aggs_and_accums, j==0)

            if j < ncols - 1:
                aa_stage_2_clear(aggs_and_accums)

        aa_stage_2_copy_back(aggs_and_accums)


    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                    plot_start, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = cuda.local.array(8, nb_types.float64) if antialias else None
        i, j = cuda.grid(2)
        if i < xs.shape[0] - 1 and j < xs.shape[1]:
            perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                plot_start, xs, ys, buffer, *aggs_and_cols)

    if use_2_stage_agg:
        return extend_cpu_antialias_2agg, extend_cuda
    else:
        return extend_cpu, extend_cuda


def _build_extend_line_axis1_none_constant(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs):
    if antialias_stage_2_funcs is not None:
        aa_stage_2_accumulate, aa_stage_2_clear, aa_stage_2_copy_back = antialias_stage_2_funcs
    use_2_stage_agg = antialias_stage_2_funcs is not None

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_line(
            i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys, buffer, *aggs_and_cols
    ):
        x0 = xs[i, j]
        y0 = ys[i, j]
        x1 = xs[i, j + 1]
        y1 = ys[i, j + 1]
        segment_start = (
                (j == 0) or isnull(xs[i, j - 1]) or isnull(ys[i, j - 1])
        )

        segment_end = (j == xs.shape[1]-2) or isnull(xs[i, j+2]) or isnull(ys[i, j+2])

        if segment_start or use_2_stage_agg:
            xm = 0.0
            ym = 0.0
        else:
            xm = xs[i, j-1]
            ym = ys[i, j-1]

        draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                     segment_start, segment_end, x0, x1, y0, y1,
                     xm, ym, buffer, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        ncols = xs.shape[1]
        for i in range(xs.shape[0]):
            for j in range(ncols - 1):
                perform_extend_line(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys, buffer, *aggs_and_cols
                )

    def extend_cpu_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                                  antialias_stage_2, *aggs_and_cols):
        n_aggs = len(antialias_stage_2[0])
        aggs_and_accums = tuple((agg, agg.copy()) for agg in aggs_and_cols[:n_aggs])
        cpu_antialias_2agg_impl(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                                antialias_stage_2, aggs_and_accums, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def cpu_antialias_2agg_impl(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                                antialias_stage_2, aggs_and_accums, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None

        ncols = xs.shape[1]
        for i in range(xs.shape[0]):
            for j in range(ncols - 1):
                perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                    xs, ys, buffer, *aggs_and_cols)

            if xs.shape[0] == 1:
                return

            aa_stage_2_accumulate(aggs_and_accums, i==0)

            if i < xs.shape[0] - 1:
                aa_stage_2_clear(aggs_and_accums)

        aa_stage_2_copy_back(aggs_and_accums)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = cuda.local.array(8, nb_types.float64) if antialias else None
        i, j = cuda.grid(2)
        if i < xs.shape[0] and j < xs.shape[1] - 1:
            perform_extend_line(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                buffer, *aggs_and_cols
            )

    if use_2_stage_agg:
        return extend_cpu_antialias_2agg, extend_cuda
    else:
        return extend_cpu, extend_cuda


def _build_extend_line_axis1_x_constant(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs):
    if antialias_stage_2_funcs is not None:
        aa_stage_2_accumulate, aa_stage_2_clear, aa_stage_2_copy_back = antialias_stage_2_funcs
    use_2_stage_agg = antialias_stage_2_funcs is not None

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_line(
            i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, buffer, *aggs_and_cols
    ):
        x0 = xs[j]
        y0 = ys[i, j]
        x1 = xs[j + 1]
        y1 = ys[i, j + 1]

        segment_start = (
                (j == 0) or isnull(xs[j - 1]) or isnull(ys[i, j - 1])
        )

        segment_end = (j == len(xs)-2) or isnull(xs[j+2]) or isnull(ys[i, j+2])

        if segment_start or use_2_stage_agg:
            xm = 0.0
            ym = 0.0
        else:
            xm = xs[j-1]
            ym = ys[i, j-1]

        draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                     segment_start, segment_end, x0, x1, y0, y1,
                     xm, ym, buffer, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        ncols = ys.shape[1]
        for i in range(ys.shape[0]):
            for j in range(ncols - 1):
                perform_extend_line(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, buffer, *aggs_and_cols
                )

    def extend_cpu_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                                  antialias_stage_2, *aggs_and_cols):
        n_aggs = len(antialias_stage_2[0])
        aggs_and_accums = tuple((agg, agg.copy()) for agg in aggs_and_cols[:n_aggs])
        cpu_antialias_2agg_impl(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                                antialias_stage_2, aggs_and_accums, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def cpu_antialias_2agg_impl(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                                antialias_stage_2, aggs_and_accums, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None

        ncols = ys.shape[1]
        for i in range(ys.shape[0]):
            for j in range(ncols - 1):
                perform_extend_line(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                    buffer, *aggs_and_cols
                )

            if ys.shape[0] == 1:
                return

            aa_stage_2_accumulate(aggs_and_accums, i==0)

            if i < ys.shape[0] - 1:
                aa_stage_2_clear(aggs_and_accums)

        aa_stage_2_copy_back(aggs_and_accums)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = cuda.local.array(8, nb_types.float64) if antialias else None
        i, j = cuda.grid(2)
        if i < ys.shape[0] and j < ys.shape[1] - 1:
            perform_extend_line(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                buffer, *aggs_and_cols
            )

    if use_2_stage_agg:
        return extend_cpu_antialias_2agg, extend_cuda
    else:
        return extend_cpu, extend_cuda


def _build_extend_line_axis1_y_constant(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs):
    if antialias_stage_2_funcs is not None:
        aa_stage_2_accumulate, aa_stage_2_clear, aa_stage_2_copy_back = antialias_stage_2_funcs
    use_2_stage_agg = antialias_stage_2_funcs is not None

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_line(
            i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, buffer, *aggs_and_cols
    ):
        x0 = xs[i, j]
        y0 = ys[j]
        x1 = xs[i, j + 1]
        y1 = ys[j + 1]

        segment_start = (
                (j == 0) or isnull(xs[i, j - 1]) or isnull(ys[j - 1])
        )

        segment_end = (j == len(ys)-2) or isnull(xs[i, j+2]) or isnull(ys[j+2])

        if segment_start or use_2_stage_agg:
            xm = 0.0
            ym = 0.0
        else:
            xm = xs[i, j-1]
            ym = ys[j-1]

        draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                     segment_start, segment_end, x0, x1, y0, y1,
                     xm, ym, buffer, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        ncols = xs.shape[1]
        for i in range(xs.shape[0]):
            for j in range(ncols - 1):
                perform_extend_line(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys, buffer, *aggs_and_cols
                )

    def extend_cpu_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                                  antialias_stage_2, *aggs_and_cols):
        n_aggs = len(antialias_stage_2[0])
        aggs_and_accums = tuple((agg, agg.copy()) for agg in aggs_and_cols[:n_aggs])
        cpu_antialias_2agg_impl(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                                antialias_stage_2, aggs_and_accums, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def cpu_antialias_2agg_impl(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                                antialias_stage_2, aggs_and_accums, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None

        ncols = xs.shape[1]
        for i in range(xs.shape[0]):

            for j in range(ncols - 1):
                perform_extend_line(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys, buffer, *aggs_and_cols
                )

            if xs.shape[0] == 1:
                return

            aa_stage_2_accumulate(aggs_and_accums, i==0)

            if i < xs.shape[0] - 1:
                aa_stage_2_clear(aggs_and_accums)

        aa_stage_2_copy_back(aggs_and_accums)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = cuda.local.array(8, nb_types.float64) if antialias else None
        i, j = cuda.grid(2)
        if i < xs.shape[0] and j < xs.shape[1] - 1:
            perform_extend_line(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                xs, ys, buffer, *aggs_and_cols
            )

    if use_2_stage_agg:
        return extend_cpu_antialias_2agg, extend_cuda
    else:
        return extend_cpu, extend_cuda


def _build_extend_line_axis1_ragged(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs):
    if antialias_stage_2_funcs is not None:
        aa_stage_2_accumulate, aa_stage_2_clear, aa_stage_2_copy_back = antialias_stage_2_funcs
    use_2_stage_agg = antialias_stage_2_funcs is not None

    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols
    ):
        x_start_i = xs.start_indices
        x_flat = xs.flat_array

        y_start_i = ys.start_indices
        y_flat = ys.flat_array

        extend_cpu_numba(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x_start_i, x_flat, y_start_i, y_flat, antialias_stage_2, *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x_start_i, x_flat, y_start_i, y_flat, antialias_stage_2, *aggs_and_cols
    ):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        nrows = len(x_start_i)
        x_flat_len = len(x_flat)
        y_flat_len = len(y_flat)

        for i in range(nrows):
            # Get x index range
            x_start_index = x_start_i[i]
            x_stop_index = (x_start_i[i + 1]
                            if i < nrows - 1
                            else x_flat_len)

            # Get y index range
            y_start_index = y_start_i[i]
            y_stop_index = (y_start_i[i + 1]
                            if i < nrows - 1
                            else y_flat_len)

            # Find line segment length as shorter of the two segments
            segment_len = min(x_stop_index - x_start_index,
                              y_stop_index - y_start_index)

            for j in range(segment_len - 1):

                x0 = x_flat[x_start_index + j]
                y0 = y_flat[y_start_index + j]
                x1 = x_flat[x_start_index + j + 1]
                y1 = y_flat[y_start_index + j + 1]

                segment_start = (
                        (j == 0) or
                        isnull(x_flat[x_start_index + j - 1]) or
                        isnull(y_flat[y_start_index + j - 1])
                )

                segment_end = (
                        (j == segment_len-2) or
                        isnull(x_flat[x_start_index + j + 2]) or
                        isnull(y_flat[y_start_index + j + 2])
                )

                if segment_start or use_2_stage_agg:
                    xm = 0.0
                    ym = 0.0
                else:
                    xm = x_flat[x_start_index + j - 1]
                    ym = y_flat[y_start_index + j - 1]

                draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                             segment_start, segment_end, x0, x1, y0, y1,
                             xm, ym, buffer, *aggs_and_cols)

    def extend_cpu_antialias_2agg(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols
    ):
        x_start_i = xs.start_indices
        x_flat = xs.flat_array

        y_start_i = ys.start_indices
        y_flat = ys.flat_array

        n_aggs = len(antialias_stage_2[0])
        aggs_and_accums = tuple((agg, agg.copy()) for agg in aggs_and_cols[:n_aggs])

        extend_cpu_numba_antialias_2agg(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, x_start_i, x_flat,
            y_start_i, y_flat, antialias_stage_2, aggs_and_accums, *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba_antialias_2agg(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, x_start_i, x_flat,
            y_start_i, y_flat, antialias_stage_2, aggs_and_accums, *aggs_and_cols
    ):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None

        nrows = len(x_start_i)
        x_flat_len = len(x_flat)
        y_flat_len = len(y_flat)

        for i in range(nrows):
            # Get x index range
            x_start_index = x_start_i[i]
            x_stop_index = (x_start_i[i + 1]
                            if i < nrows - 1
                            else x_flat_len)

            # Get y index range
            y_start_index = y_start_i[i]
            y_stop_index = (y_start_i[i + 1]
                            if i < nrows - 1
                            else y_flat_len)

            # Find line segment length as shorter of the two segments
            segment_len = min(x_stop_index - x_start_index,
                              y_stop_index - y_start_index)

            for j in range(segment_len - 1):

                x0 = x_flat[x_start_index + j]
                y0 = y_flat[y_start_index + j]
                x1 = x_flat[x_start_index + j + 1]
                y1 = y_flat[y_start_index + j + 1]

                segment_start = (
                        (j == 0) or
                        isnull(x_flat[x_start_index + j - 1]) or
                        isnull(y_flat[y_start_index + j - 1])
                )

                segment_end = (
                        (j == segment_len-2) or
                        isnull(x_flat[x_start_index + j + 2]) or
                        isnull(y_flat[y_start_index + j + 2])
                )

                draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                             segment_start, segment_end, x0, x1, y0, y1,
                             0.0, 0.0, buffer, *aggs_and_cols)

            if nrows == 1:
                return

            aa_stage_2_accumulate(aggs_and_accums, i==0)

            if i < nrows - 1:
                aa_stage_2_clear(aggs_and_accums)

        aa_stage_2_copy_back(aggs_and_accums)

    if use_2_stage_agg:
        return extend_cpu_antialias_2agg
    else:
        return extend_cpu


def _build_extend_line_axis1_geometry(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs):
    if antialias_stage_2_funcs is not None:
        aa_stage_2_accumulate, aa_stage_2_clear, aa_stage_2_copy_back = antialias_stage_2_funcs
    use_2_stage_agg = antialias_stage_2_funcs is not None

    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            geometry, closed_rings, antialias_stage_2, *aggs_and_cols
    ):
        values = geometry.buffer_values
        missing = geometry.isna()
        offsets = geometry.buffer_offsets

        if len(offsets) == 2:
            # MultiLineArray
            offsets0, offsets1 = offsets
        else:
            # LineArray
            offsets1 = offsets[0]
            offsets0 = np.arange(len(offsets1))

        if geometry._sindex is not None:
            # Compute indices of potentially intersecting polygons using
            # geometry's R-tree if there is one
            eligible_inds = geometry.sindex.intersects((xmin, ymin, xmax, ymax))
        else:
            # Otherwise, process all indices
            eligible_inds = np.arange(0, len(geometry), dtype='uint32')

        extend_cpu_numba(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            values, missing, offsets0, offsets1, eligible_inds,
            closed_rings, antialias_stage_2, *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            values, missing, offsets0, offsets1, eligible_inds,
            closed_rings, antialias_stage_2, *aggs_and_cols
    ):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        for i in eligible_inds:
            if missing[i]:
                continue

            start0 = offsets0[i]
            stop0 = offsets0[i + 1]

            for j in range(start0, stop0):
                start1 = offsets1[j]
                stop1 = offsets1[j + 1]

                for k in range(start1, stop1 - 2, 2):
                    x0 = values[k]
                    if not np.isfinite(x0):
                        continue

                    y0 = values[k + 1]
                    if not np.isfinite(y0):
                        continue

                    x1 = values[k + 2]
                    if not np.isfinite(x1):
                        continue

                    y1 = values[k + 3]
                    if not np.isfinite(y1):
                        continue

                    segment_start = (
                            (k == start1 and not closed_rings) or
                            (k > start1 and
                             (not np.isfinite(values[k - 2]) or not np.isfinite(values[k - 1])))
                    )

                    segment_end = (
                            (not closed_rings and k == stop1-4) or
                            (k < stop1-4 and
                             (not np.isfinite(values[k + 4]) or not np.isfinite(values[k + 5])))
                    )

                    if segment_start or use_2_stage_agg:
                        xm = 0.0
                        ym = 0.0
                    elif k == start1 and closed_rings:
                        xm = values[stop1-4]
                        ym = values[stop1-3]
                    else:
                        xm = values[k-2]
                        ym = values[k-1]

                    draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                 segment_start, segment_end, x0, x1, y0, y1,
                                 xm, ym, buffer, *aggs_and_cols)

    def extend_cpu_antialias_2agg(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            geometry, closed_rings, antialias_stage_2, *aggs_and_cols
    ):
        values = geometry.buffer_values
        missing = geometry.isna()
        offsets = geometry.buffer_offsets

        if len(offsets) == 2:
            # MultiLineArray
            offsets0, offsets1 = offsets
        else:
            # LineArray
            offsets1 = offsets[0]
            offsets0 = np.arange(len(offsets1))

        if geometry._sindex is not None:
            # Compute indices of potentially intersecting polygons using
            # geometry's R-tree if there is one
            eligible_inds = geometry.sindex.intersects((xmin, ymin, xmax, ymax))
        else:
            # Otherwise, process all indices
            eligible_inds = np.arange(0, len(geometry), dtype='uint32')

        n_aggs = len(antialias_stage_2[0])
        aggs_and_accums = tuple((agg, agg.copy()) for agg in aggs_and_cols[:n_aggs])

        extend_cpu_numba_antialias_2agg(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            values, missing, offsets0, offsets1, eligible_inds,
            closed_rings, antialias_stage_2, aggs_and_accums, *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba_antialias_2agg(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            values, missing, offsets0, offsets1, eligible_inds,
            closed_rings, antialias_stage_2, aggs_and_accums, *aggs_and_cols
    ):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None

        first_pass = True
        for i in eligible_inds:
            if missing[i]:
                continue

            start0 = offsets0[i]
            stop0 = offsets0[i + 1]

            for j in range(start0, stop0):
                start1 = offsets1[j]
                stop1 = offsets1[j + 1]

                for k in range(start1, stop1 - 2, 2):
                    x0 = values[k]
                    if not np.isfinite(x0):
                        continue

                    y0 = values[k + 1]
                    if not np.isfinite(y0):
                        continue

                    x1 = values[k + 2]
                    if not np.isfinite(x1):
                        continue

                    y1 = values[k + 3]
                    if not np.isfinite(y1):
                        continue

                    segment_start = (
                            (k == start1 and not closed_rings) or
                            (k > start1 and
                             (not np.isfinite(values[k - 2]) or not np.isfinite(values[k - 1])))
                    )

                    segment_end = (
                            (not closed_rings and k == stop1-4) or
                            (k < stop1-4 and
                             (not np.isfinite(values[k + 4]) or not np.isfinite(values[k + 5])))
                    )

                    draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                 segment_start, segment_end, x0, x1, y0, y1,
                                 0.0, 0.0, buffer, *aggs_and_cols)

            aa_stage_2_accumulate(aggs_and_accums, first_pass)
            first_pass = False
            aa_stage_2_clear(aggs_and_accums)

        aa_stage_2_copy_back(aggs_and_accums)

    if use_2_stage_agg:
        return extend_cpu_antialias_2agg
    else:
        return extend_cpu
