from __future__ import absolute_import, division
import numpy as np
from toolz import memoize

from datashader.glyphs.points import _PointLike
from datashader.utils import isreal, ngjit


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
        draw_line = _build_draw_line(append, expand_aggs_and_cols)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_line = _build_extend_line_axis0(
            draw_line, map_onto_pixel, expand_aggs_and_cols
        )
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = df[x_name].values
            ys = df[y_name].values
            cols = aggs + info(df)
            # line may be clipped, then mapped to pixels
            extend_line(vt, bounds, xs, ys, plot_start, *cols)

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
        bounds_list = [self._compute_x_bounds(df[x].values)
                       for x in self.x]
        mins, maxes = zip(*bounds_list)
        return self.maybe_expand_bounds((min(mins), max(maxes)))

    def compute_y_bounds(self, df):
        bounds_list = [self._compute_y_bounds(df[y].values)
                       for y in self.y]
        mins, maxes = zip(*bounds_list)
        return self.maybe_expand_bounds((min(mins), max(maxes)))

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin([np.nanmin(df[c].values) for c in self.x]),
            np.nanmax([np.nanmax(df[c].values) for c in self.x]),
            np.nanmin([np.nanmin(df[c].values) for c in self.y]),
            np.nanmax([np.nanmax(df[c].values) for c in self.y])]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, 2]), np.nanmax(r[:, 3])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_line = _build_draw_line(append, expand_aggs_and_cols)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_line = _build_extend_line_axis0_multi(
            draw_line, map_onto_pixel, expand_aggs_and_cols
        )
        x_names = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = tuple(df[x_name].values for x_name in x_names)
            ys = tuple(df[y_name].values for y_name in y_names)

            cols = aggs + info(df)
            # line may be clipped, then mapped to pixels
            extend_line(vt, bounds, xs, ys, plot_start, *cols)

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

        bounds_list = [self._compute_x_bounds(xcol.values) for xcol in xs]
        mins, maxes = zip(*bounds_list)

        return self.maybe_expand_bounds((min(mins), max(maxes)))

    def compute_y_bounds(self, df):
        ys = tuple(df[ylabel] for ylabel in self.y)

        bounds_list = [self._compute_y_bounds(ycol.values) for ycol in ys]
        mins, maxes = zip(*bounds_list)

        return self.maybe_expand_bounds((min(mins), max(maxes)))

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin([np.nanmin(df[c].values) for c in self.x]),
            np.nanmax([np.nanmax(df[c].values) for c in self.x]),
            np.nanmin([np.nanmin(df[c].values) for c in self.y]),
            np.nanmax([np.nanmax(df[c].values) for c in self.y])]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, 2]), np.nanmax(r[:, 3])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_line = _build_draw_line(append, expand_aggs_and_cols)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_lines_xy = _build_extend_line_axis1_none_constant(
            draw_line, map_onto_pixel, expand_aggs_and_cols
        )
        x_names = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = tuple(df[x_name].values for x_name in x_names)
            ys = tuple(df[y_name].values for y_name in y_names)

            cols = aggs + info(df)
            # line may be clipped, then mapped to pixels
            extend_lines_xy(vt, bounds, xs, ys, plot_start, *cols)

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
            np.nanmin([np.nanmin(df[c].values) for c in self.y]),
            np.nanmax([np.nanmax(df[c].values) for c in self.y])]]
        )).compute()

        y_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])

        return (self.compute_x_bounds(),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_line = _build_draw_line(append, expand_aggs_and_cols)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_lines = _build_extend_line_axis1_x_constant(
            draw_line, map_onto_pixel, expand_aggs_and_cols
        )

        x_values = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            ys = tuple(df[y_name].values for y_name in y_names)

            cols = aggs + info(df)
            # line may be clipped, then mapped to pixels
            extend_lines(vt, bounds, x_values, ys, plot_start, *cols)

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
            np.nanmin([np.nanmin(df[c].values) for c in self.x]),
            np.nanmax([np.nanmax(df[c].values) for c in self.x])]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])

        return (self.maybe_expand_bounds(x_extents),
                self.compute_y_bounds())

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_line = _build_draw_line(append, expand_aggs_and_cols)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_lines = _build_extend_line_axis1_y_constant(
            draw_line, map_onto_pixel, expand_aggs_and_cols
        )

        x_names = self.x
        y_values = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = tuple(df[x_name].values for x_name in x_names)

            cols = aggs + info(df)
            # line may be clipped, then mapped to pixels
            extend_lines(vt, bounds, xs, y_values, plot_start, *cols)

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
        bounds = self._compute_x_bounds(df[self.x].array.flat_array)
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds = self._compute_y_bounds(df[self.y].array.flat_array)
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin(df[self.x].array.flat_array),
            np.nanmax(df[self.x].array.flat_array),
            np.nanmin(df[self.y].array.flat_array),
            np.nanmax(df[self.y].array.flat_array)]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, 2]), np.nanmax(r[:, 3])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_line = _build_draw_line(append, expand_aggs_and_cols)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_lines_ragged = _build_extend_line_axis1_ragged(
            draw_line, map_onto_pixel, expand_aggs_and_cols
        )
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = df[x_name].array
            ys = df[y_name].array

            cols = aggs + info(df)
            # line may be clipped, then mapped to pixels
            extend_lines_ragged(vt, bounds, xs, ys, plot_start, *cols)

        return extend


def _build_map_onto_pixel_for_line(x_mapper, y_mapper):
    @ngjit
    def map_onto_pixel(vt, bounds, x, y):
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
        sx, tx, sy, ty = vt
        xmax, ymax = bounds[1], bounds[3]
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


def _build_draw_line(append, expand_aggs_and_cols):
    """Specialize a line plotting kernel for a given append/axis combination"""
    @ngjit
    @expand_aggs_and_cols
    def draw_line(x0i, y0i, x1i, y1i, i, plot_start, clipped, *aggs_and_cols):
        """Draw a line using Bresenham's algorithm

        This method plots a line segment with integer coordinates onto a pixel
        grid. The vertices are assumed to have already been scaled, transformed,
        and clipped within the bounds.

        The following algorithm is the more general Bresenham's algorithm that
        works with both float and integer coordinates. A future performance
        improvement would replace this algorithm with the integer-specific one.
        """
        dx = x1i - x0i
        ix = (dx > 0) - (dx < 0)
        dx = abs(dx) * 2

        dy = y1i - y0i
        iy = (dy > 0) - (dy < 0)
        dy = abs(dy) * 2

        if plot_start:
            append(i, x0i, y0i, *aggs_and_cols)

        if dx >= dy:
            # If vertices weren't clipped and are concurrent in integer space,
            # call append and return, as the second vertex won't be hit below.
            if not clipped and not (dx | dy):
                append(i, x0i, y0i, *aggs_and_cols)
                return
            error = 2*dy - dx
            while x0i != x1i:
                if error >= 0 and (error or ix > 0):
                    error -= 2 * dx
                    y0i += iy
                error += 2 * dy
                x0i += ix
                append(i, x0i, y0i, *aggs_and_cols)
        else:
            error = 2*dx - dy
            while y0i != y1i:
                if error >= 0 and (error or iy > 0):
                    error -= 2 * dy
                    x0i += ix
                error += 2 * dx
                y0i += iy
                append(i, x0i, y0i, *aggs_and_cols)

    return draw_line


@ngjit
def _outside_bounds(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    if x0 < xmin and x1 < xmin:
        return True
    if x0 > xmax and x1 > xmax:
        return True
    if y0 < ymin and y1 < ymin:
        return True
    return y0 > ymax and y1 > ymax


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
def _skip_or_clip(x0, x1, y0, y1, bounds, plot_start):
    xmin, xmax, ymin, ymax = bounds
    skip = False

    # If any of the coordinates are NaN, there's a discontinuity.
    # Skip the entire segment.
    if np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(
            y1):
        plot_start = True
        skip = True

    # Use Liang-Barsky (1992) to clip the segment to a bounding box
    if _outside_bounds(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
        plot_start = True
        skip = True

    clipped = False
    t0, t1 = 0, 1
    dx = x1 - x0
    t0, t1, accept = _clipt(-dx, x0 - xmin, t0, t1)
    if not accept:
        skip = True

    t0, t1, accept = _clipt(dx, xmax - x0, t0, t1)
    if not accept:
        skip = True

    dy = y1 - y0
    t0, t1, accept = _clipt(-dy, y0 - ymin, t0, t1)
    if not accept:
        skip = True

    t0, t1, accept = _clipt(dy, ymax - y0, t0, t1)
    if not accept:
        skip = True

    if t1 < 1:
        clipped = True
        x1 = x0 + t1 * dx
        y1 = y0 + t1 * dy

    if t0 > 0:
        # If x0 is clipped, we need to plot the new start
        clipped = True
        plot_start = True
        x0 = x0 + t0 * dx
        y0 = y0 + t0 * dy

    return x0, x1, y0, y1, skip, clipped, plot_start


def _build_extend_line_axis0(draw_line, map_onto_pixel, expand_aggs_and_cols):
    @ngjit
    @expand_aggs_and_cols
    def extend_line(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        """Aggregate along a line formed by ``xs`` and ``ys``"""
        nrows = xs.shape[0]
        i = 0
        while i < nrows - 1:
            x0 = xs[i]
            y0 = ys[i]
            x1 = xs[i + 1]
            y1 = ys[i + 1]

            x0, x1, y0, y1, skip, clipped, plot_start = \
                _skip_or_clip(x0, x1, y0, y1, bounds, plot_start)

            if not skip:
                x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                x1i, y1i = map_onto_pixel(vt, bounds, x1, y1)
                draw_line(x0i, y0i, x1i, y1i, i, plot_start, clipped, *aggs_and_cols)
                plot_start = False
            i += 1

    return extend_line


def _build_extend_line_axis0_multi(draw_line, map_onto_pixel, expand_aggs_and_cols):
    @ngjit
    @expand_aggs_and_cols
    def extend_line(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        """Aggregate along a line formed by ``xs`` and ``ys``"""
        nrows = xs[0].shape[0]
        ncols = len(xs)
        orig_plot_start = plot_start

        j = 0
        while j < ncols:
            plot_start = orig_plot_start
            i = 0
            while i < nrows - 1:
                x0 = xs[j][i]
                y0 = ys[j][i]
                x1 = xs[j][i + 1]
                y1 = ys[j][i + 1]

                x0, x1, y0, y1, skip, clipped, plot_start = \
                    _skip_or_clip(x0, x1, y0, y1, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    x1i, y1i = map_onto_pixel(vt, bounds, x1, y1)
                    draw_line(x0i, y0i, x1i, y1i, i, plot_start, clipped, *aggs_and_cols)
                    plot_start = False
                i += 1
            j += 1

    return extend_line


def _build_extend_line_axis1_none_constant(
        draw_line, map_onto_pixel, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def extend_line(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        """
        here xs and ys are tuples of arrays and non-empty
        """
        nrows = xs[0].shape[0]
        ncols = len(xs)

        i = 0
        while i < nrows:
            plot_start = True
            j = 0
            while j < ncols - 1:
                x0 = xs[j][i]
                y0 = ys[j][i]
                x1 = xs[j + 1][i]
                y1 = ys[j + 1][i]

                x0, x1, y0, y1, skip, clipped, plot_start = \
                    _skip_or_clip(x0, x1, y0, y1, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    x1i, y1i = map_onto_pixel(vt, bounds, x1, y1)
                    draw_line(x0i, y0i, x1i, y1i, i, plot_start, clipped,
                              *aggs_and_cols)
                    plot_start = False
                j += 1
            i += 1

    return extend_line


def _build_extend_line_axis1_x_constant(
        draw_line, map_onto_pixel, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def extend_line(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        """
        here xs and ys are tuples of arrays and non-empty
        """
        nrows = ys[0].shape[0]
        ncols = len(ys)

        i = 0
        while i < nrows:
            plot_start = True
            j = 0
            while j < ncols - 1:
                x0 = xs[j]
                y0 = ys[j][i]
                x1 = xs[j+1]
                y1 = ys[j+1][i]

                x0, x1, y0, y1, skip, clipped, plot_start = \
                    _skip_or_clip(x0, x1, y0, y1, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    x1i, y1i = map_onto_pixel(vt, bounds, x1, y1)
                    draw_line(x0i, y0i, x1i, y1i, i, plot_start, clipped,
                              *aggs_and_cols)
                    plot_start = False
                j += 1
            i += 1

    return extend_line


def _build_extend_line_axis1_y_constant(
        draw_line, map_onto_pixel, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def extend_line(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        """
        here xs and ys are tuples of arrays and non-empty
        """
        nrows = xs[0].shape[0]
        ncols = len(xs)

        i = 0
        while i < nrows:
            plot_start = True
            j = 0
            while j < ncols - 1:
                x0 = xs[j][i]
                y0 = ys[j]
                x1 = xs[j + 1][i]
                y1 = ys[j + 1]

                x0, x1, y0, y1, skip, clipped, plot_start = \
                    _skip_or_clip(x0, x1, y0, y1, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    x1i, y1i = map_onto_pixel(vt, bounds, x1, y1)
                    draw_line(x0i, y0i, x1i, y1i, i, plot_start, clipped,
                              *aggs_and_cols)
                    plot_start = False
                j += 1
            i += 1

    return extend_line


def _build_extend_line_axis1_ragged(
        draw_line, map_onto_pixel, expand_aggs_and_cols
):

    def extend_line(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        x_start_indices = xs.start_indices
        x_flat_array = xs.flat_array

        y_start_indices = ys.start_indices
        y_flat_array = ys.flat_array

        perform_extend_lines_ragged(vt,
                                    bounds,
                                    x_start_indices,
                                    x_flat_array,
                                    y_start_indices,
                                    y_flat_array,
                                    plot_start,
                                    *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_lines_ragged(vt,
                                    bounds,
                                    x_start_indices,
                                    x_flat_array,
                                    y_start_indices,
                                    y_flat_array,
                                    plot_start,
                                    *aggs_and_cols):

        nrows = len(x_start_indices)
        x_flat_len = len(x_flat_array)
        y_flat_len = len(y_flat_array)

        i = 0
        while i < nrows:
            plot_start = True

            # Get x index range
            x_start_index = x_start_indices[i]
            x_stop_index = (x_start_indices[i + 1]
                            if i < nrows - 1
                            else x_flat_len)

            # Get y index range
            y_start_index = y_start_indices[i]
            y_stop_index = (y_start_indices[i + 1]
                            if i < nrows - 1
                            else y_flat_len)

            # Find line segment length as shorter of the two segments
            segment_len = min(x_stop_index - x_start_index,
                              y_stop_index - y_start_index)

            j = 0
            while j < segment_len - 1:

                x0 = x_flat_array[x_start_index + j]
                y0 = y_flat_array[y_start_index + j]
                x1 = x_flat_array[x_start_index + j + 1]
                y1 = y_flat_array[y_start_index + j + 1]

                x0, x1, y0, y1, skip, clipped, plot_start = \
                    _skip_or_clip(x0, x1, y0, y1, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    x1i, y1i = map_onto_pixel(vt, bounds, x1, y1)
                    draw_line(x0i, y0i, x1i, y1i, i, plot_start, clipped,
                              *aggs_and_cols)
                    plot_start = False

                j += 1
            i += 1

    return extend_line
