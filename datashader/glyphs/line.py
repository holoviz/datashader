from __future__ import absolute_import, division
import numpy as np
from toolz import memoize

from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.glyphs.glyph import isnull
from datashader.utils import isreal, ngjit
from numba import cuda

try:
    import cudf
    from ..transfer_functions._cuda_utils import cuda_args
except Exception:
    cudf = None
    cuda_args = None


class LineAxis0(_PointLike):
    """A line, with vertices defined by ``x`` and ``y``.

    Parameters
    ----------
    x, y : str
        Column names for the x and y coordinates of each vertex.
    """
    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_segment = _build_draw_segment(append, map_onto_pixel, expand_aggs_and_cols)
        extend_cpu, extend_cuda = _build_extend_line_axis0(
            draw_segment, expand_aggs_and_cols
        )
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df)

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_gpu_matrix(df, x_name)
                ys = self.to_gpu_matrix(df, y_name)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df[x_name].values
                ys = df[y_name].values
                do_extend = extend_cpu

            # line may be clipped, then mapped to pixels
            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                xs, ys, plot_start, *aggs_and_cols
            )

        return extend


class LineAxis0Multi(_PointLike):
    """
    """

    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(xcol)]) for xcol in self.x]):
            raise ValueError('x columns must be real')
        elif not all([isreal(in_dshape.measure[str(ycol)]) for ycol in self.y]):
            raise ValueError('y columns must be real')

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
    def _build_extend(self, x_mapper, y_mapper, info, append):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols
        )
        extend_cpu, extend_cuda = _build_extend_line_axis0_multi(
            draw_segment, expand_aggs_and_cols
        )
        x_names = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df)

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_gpu_matrix(df, x_names)
                ys = self.to_gpu_matrix(df, y_names)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df[list(x_names)].values
                ys = df[list(y_names)].values
                do_extend = extend_cpu

            # line may be clipped, then mapped to pixels
            do_extend(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                xs, ys, plot_start, *aggs_and_cols
            )

        return extend


class LinesAxis1(_PointLike):
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
    def _build_extend(self, x_mapper, y_mapper, info, append):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols
        )
        extend_cpu, extend_cuda = _build_extend_line_axis1_none_constant(
            draw_segment, expand_aggs_and_cols
        )
        x_names = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df)

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_gpu_matrix(df, x_names)
                ys = self.to_gpu_matrix(df, y_names)
                do_extend = extend_cuda[cuda_args(xs.shape)]

            else:
                xs = df[list(x_names)].values
                ys = df[list(y_names)].values
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
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
    def _build_extend(self, x_mapper, y_mapper, info, append):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols
        )

        extend_cpu, extend_cuda = _build_extend_line_axis1_x_constant(
            draw_segment, expand_aggs_and_cols
        )

        x_values = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df)

            if cudf and isinstance(df, cudf.DataFrame):
                ys = self.to_gpu_matrix(df, y_names)
                do_extend = extend_cuda[cuda_args(ys.shape)]
            else:
                ys = df[list(y_names)].values
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                x_values, ys, *aggs_and_cols
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
    def _build_extend(self, x_mapper, y_mapper, info, append):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)

        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols
        )
        extend_cpu, extend_cuda = _build_extend_line_axis1_y_constant(
            draw_segment, expand_aggs_and_cols
        )

        x_names = self.x
        y_values = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df)

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_gpu_matrix(df, x_names)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df[list(x_names)].values
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                xs, y_values, *aggs_and_cols
            )

        return extend


class LinesAxis1Ragged(_PointLike):
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
    def _build_extend(self, x_mapper, y_mapper, info, append):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols
        )

        extend_cpu = _build_extend_line_axis1_ragged(
            draw_segment, expand_aggs_and_cols
        )
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds

            xs = df[x_name].array
            ys = df[y_name].array

            aggs_and_cols = aggs + info(df)
            # line may be clipped, then mapped to pixels
            extend_cpu(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                xs, ys, *aggs_and_cols
            )

        return extend


class LineAxis1Geometry(_GeometryLike):

    @property
    def geom_dtypes(self):
        from spatialpandas.geometry import (
            LineDtype, MultiLineDtype, RingDtype, PolygonDtype,
            MultiPolygonDtype
        )
        return (LineDtype, MultiLineDtype, RingDtype,
                PolygonDtype, MultiPolygonDtype)

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        from spatialpandas.geometry import (
            PolygonArray, MultiPolygonArray, RingArray
        )
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_segment = _build_draw_segment(
            append, map_onto_pixel, expand_aggs_and_cols
        )

        perform_extend_cpu = _build_extend_line_axis1_geometry(
            draw_segment, expand_aggs_and_cols
        )
        geometry_name = self.geometry

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df)
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
                geom_array, closed_rings, *aggs_and_cols
            )

        return extend


def _build_map_onto_pixel_for_line(x_mapper, y_mapper):
    @ngjit
    def map_onto_pixel(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x, y):
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

    return map_onto_pixel


def _build_draw_segment(append, map_onto_pixel, expand_aggs_and_cols):
    """Specialize a line plotting kernel for a given append/axis combination"""
    @ngjit
    @expand_aggs_and_cols
    def draw_segment(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start,
            x0, x1, y0, y1, *aggs_and_cols
    ):
        """Draw a line segment using Bresenham's algorithm
        This method plots a line segment with integer coordinates onto a pixel
        grid.
        """
        skip = False

        # If any of the coordinates are NaN, there's a discontinuity.
        # Skip the entire segment.
        if isnull(x0) or isnull(y0) or isnull(x1) or isnull(y1):
            skip = True

        # Use Liang-Barsky (1992) to clip the segment to a bounding box
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

        segment_start = segment_start or clipped_start
        if not skip:
            x0i, y0i = map_onto_pixel(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0, y0
            )
            x1i, y1i = map_onto_pixel(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax, x1, y1
            )
            clipped = clipped_start or clipped_end

            dx = x1i - x0i
            ix = (dx > 0) - (dx < 0)
            dx = abs(dx) * 2

            dy = y1i - y0i
            iy = (dy > 0) - (dy < 0)
            dy = abs(dy) * 2

            # If vertices weren't clipped and are concurrent in integer space,
            # call append and return, so that the second vertex won't be hit below.
            if not clipped and not (dx | dy):
                append(i, x0i, y0i, *aggs_and_cols)
                return

            if segment_start:
                append(i, x0i, y0i, *aggs_and_cols)

            if dx >= dy:
                error = 2 * dy - dx
                while x0i != x1i:
                    if error >= 0 and (error or ix > 0):
                        error -= 2 * dx
                        y0i += iy
                    error += 2 * dy
                    x0i += ix
                    append(i, x0i, y0i, *aggs_and_cols)
            else:
                error = 2 * dx - dy
                while y0i != y1i:
                    if error >= 0 and (error or iy > 0):
                        error -= 2 * dy
                        x0i += ix
                    error += 2 * dx
                    y0i += iy
                    append(i, x0i, y0i, *aggs_and_cols)

    return draw_segment


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


def _build_extend_line_axis0(draw_segment, expand_aggs_and_cols):

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_line(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                             plot_start, xs, ys, *aggs_and_cols):
        x0 = xs[i]
        y0 = ys[i]
        x1 = xs[i + 1]
        y1 = ys[i + 1]
        segment_start = (plot_start if i == 0 else
                         (isnull(xs[i - 1]) or isnull(ys[i - 1])))

        draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                     segment_start, x0, x1, y0, y1, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                   xs, ys, plot_start, *aggs_and_cols
    ):
        """Aggregate along a line formed by ``xs`` and ``ys``"""
        nrows = xs.shape[0]
        for i in range(nrows - 1):
            perform_extend_line(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                plot_start, xs, ys, *aggs_and_cols)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys, plot_start, *aggs_and_cols):
        i = cuda.grid(1)
        if i < xs.shape[0] - 1:
            perform_extend_line(
                i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                plot_start, xs, ys, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_line_axis0_multi(draw_segment, expand_aggs_and_cols):

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                            plot_start, xs, ys, *aggs_and_cols):
        x0 = xs[i, j]
        y0 = ys[i, j]
        x1 = xs[i + 1, j]
        y1 = ys[i + 1, j]
        segment_start = (plot_start if i == 0 else
                         (isnull(xs[i - 1, j]) or isnull(ys[i - 1, j])))
        draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                     segment_start, x0, x1, y0, y1, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty,
            xmin, xmax, ymin, ymax,
            xs, ys, plot_start, *aggs_and_cols):
        """Aggregate along a line formed by ``xs`` and ``ys``"""
        nrows, ncols = xs.shape

        for j in range(ncols):
            for i in range(nrows - 1):
                perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                    plot_start, xs, ys, *aggs_and_cols)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                    plot_start, *aggs_and_cols):
        i, j = cuda.grid(2)
        if i < xs.shape[0] - 1 and j < xs.shape[1]:
            perform_extend_line(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                plot_start, xs, ys, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_line_axis1_none_constant(draw_segment, expand_aggs_and_cols):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend_line(
            i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys, *aggs_and_cols
    ):
        x0 = xs[i, j]
        y0 = ys[i, j]
        x1 = xs[i, j + 1]
        y1 = ys[i, j + 1]
        segment_start = (
                (j == 0) or isnull(xs[i, j - 1]) or isnull(ys[i, j - 1])
        )

        draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                     segment_start, x0, x1, y0, y1, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
    ):
        ncols = xs.shape[1]
        for i in range(xs.shape[0]):
            for j in range(ncols - 1):
                perform_extend_line(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys, *aggs_and_cols
                )

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                    *aggs_and_cols):
        i, j = cuda.grid(2)
        if i < xs.shape[0] and j < xs.shape[1] - 1:
            perform_extend_line(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_line_axis1_x_constant(
        draw_segment, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend_line(
            i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
    ):
        x0 = xs[j]
        y0 = ys[i, j]
        x1 = xs[j + 1]
        y1 = ys[i, j + 1]

        segment_start = (
                (j == 0) or isnull(xs[j - 1]) or isnull(ys[i, j - 1])
        )

        draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                     segment_start, x0, x1, y0, y1, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols):
        ncols = ys.shape[1]
        for i in range(ys.shape[0]):
            for j in range(ncols - 1):
                perform_extend_line(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
                )

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                     *aggs_and_cols):
        i, j = cuda.grid(2)
        if i < ys.shape[0] and j < ys.shape[1] - 1:
            perform_extend_line(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys,
                *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_line_axis1_y_constant(
        draw_segment, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend_line(
            i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
    ):
        x0 = xs[i, j]
        y0 = ys[j]
        x1 = xs[i, j + 1]
        y1 = ys[j + 1]

        segment_start = (
                (j == 0) or isnull(xs[i, j - 1]) or isnull(ys[j - 1])
        )

        draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                     segment_start, x0, x1, y0, y1, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty,
            xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
    ):
        ncols = xs.shape[1]
        for i in range(xs.shape[0]):
            for j in range(ncols - 1):
                perform_extend_line(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys, *aggs_and_cols
                )

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(
            sx, tx, sy, ty,
            xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
    ):
        i, j = cuda.grid(2)
        if i < xs.shape[0] and j < xs.shape[1] - 1:
            perform_extend_line(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                xs, ys, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_line_axis1_ragged(
        draw_segment, expand_aggs_and_cols
):

    def extend_cpu(
            sx, tx, sy, ty,
            xmin, xmax, ymin, ymax,
            xs, ys, *aggs_and_cols
    ):
        x_start_i = xs.start_indices
        x_flat = xs.flat_array

        y_start_i = ys.start_indices
        y_flat = ys.flat_array

        extend_cpu_numba(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x_start_i, x_flat, y_start_i, y_flat, *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x_start_i, x_flat, y_start_i, y_flat, *aggs_and_cols
    ):
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
                        isnull(y_flat[y_start_index + j] - 1)
                )

                draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                             segment_start, x0, x1, y0, y1, *aggs_and_cols)

    return extend_cpu


def _build_extend_line_axis1_geometry(
        draw_segment, expand_aggs_and_cols
):
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            geometry, closed_rings, *aggs_and_cols
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
            closed_rings, *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            values, missing, offsets0, offsets1, eligible_inds,
            closed_rings, *aggs_and_cols
    ):
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
                             not np.isfinite(values[k - 2]) or
                             not np.isfinite(values[k - 1]))
                    )

                    draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                                 segment_start, x0, x1, y0, y1, *aggs_and_cols)

    return extend_cpu
