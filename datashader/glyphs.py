from __future__ import absolute_import, division

from toolz import memoize
import numpy as np

from .utils import ngjit, isreal, Expr


class Glyph(Expr):
    """Base class for glyphs."""
    @staticmethod
    def maybe_expand_bounds(bounds):
        minval, maxval = bounds
        if not (np.isfinite(minval) and np.isfinite(maxval)):
            minval, maxval = -1.0, 1.0
        elif minval == maxval:
            minval, maxval = minval-1, minval+1
        return minval, maxval

    @staticmethod
    @ngjit
    def _compute_x_bounds(xs):
        minval = np.inf
        maxval = -np.inf
        for x in xs:
            if not np.isnan(x):
                if x < minval:
                    minval = x
                if x > maxval:
                    maxval = x

        return minval, maxval

    @staticmethod
    @ngjit
    def _compute_y_bounds(ys):
        minval = np.inf
        maxval = -np.inf
        for y in ys:
            if not np.isnan(y):
                if y < minval:
                    minval = y
                if y > maxval:
                    maxval = y

        return minval, maxval


class _PointLike(Glyph):
    """Shared methods between Point and Line"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

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
        bounds = self._compute_x_bounds(df[self.x].values)
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds = self._compute_y_bounds(df[self.y].values)
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


class _PolygonLike(_PointLike):
    """_PointLike class, with methods overridden for vertex-delimited shapes.

    Key differences from _PointLike:
        - added self.z as a list, representing vertex weights
        - constructor accepts additional kwargs:
            * weight_type (bool): Whether the weights are on vertices (True) or on the shapes (False)
            * interp (bool): Whether to interpolate (True), or to have one color per shape (False)
    """
    def __init__(self, x, y, z=None, weight_type=True, interp=True):
        super(_PolygonLike, self).__init__(x, y)
        if z is None:
            self.z = []
        else:
            self.z = z
        self.interpolate = interp
        self.weight_type = weight_type

    @property
    def inputs(self):
        return (tuple([self.x, self.y] + list(self.z)) +
                (self.weight_type, self.interpolate))

    def validate(self, in_dshape):
        for col in [self.x, self.y] + list(self.z):
            if not isreal(in_dshape.measure[str(col)]):
                raise ValueError('{} must be real'.format(col))

    def required_columns(self):
        return [self.x, self.y] + list(self.z)

    def compute_x_bounds(self, df):
        xs = df[self.x].values
        bounds = self._compute_x_bounds(xs.reshape(np.prod(xs.shape)))
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        ys = df[self.y].values
        bounds = self._compute_y_bounds(ys.reshape(np.prod(ys.shape)))
        return self.maybe_expand_bounds(bounds)


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
        def _extend(vt, bounds, xs, ys, *aggs_and_cols):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds

            def map_onto_pixel(x, y):
                """Map points onto pixel grid.

                Points falling on upper bound are mapped into previous bin.
                """
                xx = int(x_mapper(x) * sx + tx)
                yy = int(y_mapper(y) * sy + ty)
                return (xx - 1 if x == xmax else xx,
                        yy - 1 if y == ymax else yy)

            for i in range(xs.shape[0]):
                x = xs[i]
                y = ys[i]
                # points outside bounds are dropped; remainder
                # are mapped onto pixels
                if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                    xi, yi = map_onto_pixel(x, y)
                    append(i, xi, yi, *aggs_and_cols)

        def extend(aggs, df, vt, bounds):
            xs = df[x_name].values
            ys = df[y_name].values
            cols = aggs + info(df)
            _extend(vt, bounds, xs, ys, *cols)

        return extend


class LineAxis0(_PointLike):
    """A line, with vertices defined by ``x`` and ``y``.

    Parameters
    ----------
    x, y : str
        Column names for the x and y coordinates of each vertex.
    """
    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_line = _build_draw_line(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_line = _build_extend_line_axis0(draw_line, map_onto_pixel)
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
        draw_line = _build_draw_line(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_line = _build_extend_line_axis0_multi(draw_line, map_onto_pixel)
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
        draw_line = _build_draw_line(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_lines_xy = _build_extend_line_axis1_none_constant(draw_line, map_onto_pixel)
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
        draw_line = _build_draw_line(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_lines = _build_extend_line_axis1_x_constant(draw_line, map_onto_pixel)

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
        draw_line = _build_draw_line(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_lines = _build_extend_line_axis1_y_constant(draw_line, map_onto_pixel)

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
        draw_line = _build_draw_line(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_lines_ragged = _build_extend_line_axis1_ragged(draw_line, map_onto_pixel)
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = df[x_name].array
            ys = df[y_name].array

            cols = aggs + info(df)
            # line may be clipped, then mapped to pixels
            extend_lines_ragged(vt, bounds, xs, ys, plot_start, *cols)

        return extend


class _AreaToLineLike(Glyph):
    """Shared methods between Point and Line"""
    def __init__(self, x, y, y_stack):
        self.x = x
        self.y = y
        self.y_stack = y_stack

    @property
    def inputs(self):
        return (self.x, self.y, self.y_stack)

    def validate(self, in_dshape):
        if not isreal(in_dshape.measure[str(self.x)]):
            raise ValueError('x must be real')
        elif not isreal(in_dshape.measure[str(self.y)]):
            raise ValueError('y must be real')
        elif not isreal(in_dshape.measure[str(self.y_stack)]):
            raise ValueError('y_stack must be real or None')

    @property
    def x_label(self):
        return self.x

    @property
    def y_label(self):
        return self.y

    def required_columns(self):
        return self.x, self.y, self.y_stack

    def compute_x_bounds(self, df):
        bounds = self._compute_x_bounds(df[self.x].values)
        return self.maybe_expand_bounds(bounds)


class AreaToZeroAxis0(_PointLike):
    """A filled area glyph
    The area to be filled is the region from the line defined by ``x`` and
    ``y`` and the y=0 line

    Parameters
    ----------
    x, y
        Column names for the x and y coordinates of each vertex.
    """

    def compute_y_bounds(self, df):
        # Compute bounds of curve
        bounds = self._compute_y_bounds(df[self.y].values)

        # Make sure bounds include zero
        bounds = min(bounds[0], 0), max(bounds[1], 0)

        # Expand bounds if needed
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

        # Make sure y_extents include zero
        y_extents = min(y_extents[0], 0), max(y_extents[1], 0)

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_trapezoid_y = _build_draw_trapezoid_y(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_area = _build_extend_area_to_zero_axis0(
            draw_trapezoid_y, map_onto_pixel)
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = df[x_name].values
            ys = df[y_name].values
            cols = aggs + info(df)
            extend_area(vt, bounds, xs, ys, plot_start, *cols)

        return extend


class AreaToLineAxis0(_AreaToLineLike):
    """A filled area glyph
    The area to be filled is the region from the line defined by ``x`` and
    ``y[0]`` and the line defined by ``x`` and ``y[1]``.

    Parameters
    ----------
    x
        Column names for the x and y coordinates of each vertex.
    y
        List or tuple of length two containing the column names of the
        y-coordinates of the two curves that define the area region.
    """

    def compute_y_bounds(self, df):
        # Compute bounds of each curve
        bounds0 = self._compute_y_bounds(df[self.y].values)
        bounds1 = self._compute_y_bounds(df[self.y_stack].values)

        # Combine bounds from the two curves
        bounds = min(bounds0[0], bounds1[0]), max(bounds0[1], bounds1[1])

        # Expand bounds if needed
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin(df[self.x].values),
            np.nanmax(df[self.x].values),
            np.nanmin(df[self.y].values),
            np.nanmax(df[self.y].values),
            np.nanmin(df[self.y_stack].values),
            np.nanmax(df[self.y_stack].values)]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y0_extents = np.nanmin(r[:, 2]), np.nanmax(r[:, 3])
        y1_extents = np.nanmin(r[:, 4]), np.nanmax(r[:, 5])

        # Make sure y_extents include 0
        y_extents = (min(y0_extents[0], y1_extents[0]),
                     max(y0_extents[1], y1_extents[1]))

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_trapezoid_y = _build_draw_trapezoid_y(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_area = _build_extend_area_to_line_axis0(
            draw_trapezoid_y, map_onto_pixel)
        x_name = self.x
        y_name = self.y
        y_stack_name = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = df[x_name].values
            ys = df[y_name].values
            ys_stacks = df[y_stack_name].values

            cols = aggs + info(df)
            extend_area(vt, bounds, xs, ys, ys_stacks, plot_start, *cols)

        return extend


class AreaToZeroAxis0Multi(_PointLike):
    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(xcol)]) for xcol in self.x]):
            raise ValueError('x columns must be real')
        elif not all(
                [isreal(in_dshape.measure[str(ycol)]) for ycol in self.y]):
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

        # Make sure min/max range includes zero
        mn = min(0, min(mins))
        mx = max(0, max(maxes))
        return self.maybe_expand_bounds((mn, mx))

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

        # Makes sure y_extents include 0
        y_extents = min(0, y_extents[0]), max(0, y_extents[1])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_trapezoid_y = _build_draw_trapezoid_y(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_area = _build_extend_area_to_zero_axis0_multi(
            draw_trapezoid_y, map_onto_pixel)
        x_names = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = tuple(df[x_name].values for x_name in x_names)
            ys = tuple(df[y_name].values for y_name in y_names)
            cols = aggs + info(df)
            extend_area(vt, bounds, xs, ys, plot_start, *cols)

        return extend


class AreaToLineAxis0Multi(_AreaToLineLike):
    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(xcol)]) for xcol in self.x]):
            raise ValueError('x columns must be real')
        elif not all(
                [isreal(in_dshape.measure[str(ycol)]) for ycol in self.y]):
            raise ValueError('y columns must be real')
        elif not all(
                [isreal(in_dshape.measure[str(ycol)])
                 for ycol in self.y_stack]):
            raise ValueError('y_stack columns must be real')

    @property
    def x_label(self):
        return 'x'

    @property
    def y_label(self):
        return 'y'

    def required_columns(self):
        return self.x + self.y + self.y_stack

    def compute_x_bounds(self, df):
        bounds_list = [self._compute_x_bounds(df[x].values)
                       for x in self.x]
        mins, maxes = zip(*bounds_list)
        return self.maybe_expand_bounds((min(mins), max(maxes)))

    def compute_y_bounds(self, df):
        bounds_list = [self._compute_y_bounds(df[y].values)
                       for y in self.y + self.y_stack]
        mins, maxes = zip(*bounds_list)

        return self.maybe_expand_bounds((min(mins), max(maxes)))

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin([np.nanmin(df[c].values) for c in self.x]),
            np.nanmax([np.nanmax(df[c].values) for c in self.x]),
            np.nanmin([np.nanmin(df[c].values) for c in self.y]),
            np.nanmax([np.nanmax(df[c].values) for c in self.y]),
            np.nanmin([np.nanmin(df[c].values) for c in self.y_stack]),
            np.nanmax([np.nanmax(df[c].values) for c in self.y_stack]),
        ]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, [2, 4]]), np.nanmax(r[:, [3, 5]])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_trapezoid_y = _build_draw_trapezoid_y(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_area = _build_extend_area_to_line_axis0_multi(
            draw_trapezoid_y, map_onto_pixel)
        x_names = self.x
        y_names = self.y
        y_stack_names = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = tuple(df[x_name].values for x_name in x_names)
            ys = tuple(df[y_name].values for y_name in y_names)
            y_stacks = tuple(df[y_stack_name].values
                             for y_stack_name in y_stack_names)

            cols = aggs + info(df)
            extend_area(vt, bounds, xs, ys, y_stacks, plot_start, *cols)

        return extend


class AreaToZeroAxis1(_PointLike):
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

        # Make sure min/max range includes zero
        mn = min(0, min(mins))
        mx = max(0, max(maxes))

        return self.maybe_expand_bounds((mn, mx))

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

        # Makes sure y_extents include 0
        y_extents = min(0, y_extents[0]), max(0, y_extents[1])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_trapezoid_y = _build_draw_trapezoid_y(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_area = _build_extend_area_to_zero_axis1_none_constant(
            draw_trapezoid_y, map_onto_pixel)

        x_names = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = tuple(df[x_name].values for x_name in x_names)
            ys = tuple(df[y_name].values for y_name in y_names)

            cols = aggs + info(df)
            # line may be clipped, then mapped to pixels
            extend_area(vt, bounds, xs, ys, plot_start, *cols)

        return extend


class AreaToLineAxis1(_AreaToLineLike):
    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(xcol)])
                    for xcol in self.x]):
            raise ValueError('x columns must be real')
        elif not all([isreal(in_dshape.measure[str(ycol)])
                      for ycol in self.y]):
            raise ValueError('y columns must be real')
        elif not all([isreal(in_dshape.measure[str(ycol)])
                      for ycol in self.y_stack]):
            raise ValueError('y_stack columns must be real')

        unique_x_measures = set(in_dshape.measure[str(xcol)]
                                for xcol in self.x)
        if len(unique_x_measures) > 1:
            raise ValueError('x columns must have the same data type')

        unique_y_measures = set(in_dshape.measure[str(ycol)]
                                for ycol in self.y)
        if len(unique_y_measures) > 1:
            raise ValueError('y columns must have the same data type')

        unique_y_stack_measures = set(in_dshape.measure[str(ycol)]
                                      for ycol in self.y_stack)
        if len(unique_y_stack_measures) > 1:
            raise ValueError('y_stack columns must have the same data type')

    def required_columns(self):
        return self.x + self.y + self.y_stack

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
        bounds_list = [self._compute_y_bounds(df[y].values)
                       for y in self.y + self.y_stack]
        mins, maxes = zip(*bounds_list)

        return self.maybe_expand_bounds((min(mins), max(maxes)))

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin([np.nanmin(df[c].values) for c in self.x]),
            np.nanmax([np.nanmax(df[c].values) for c in self.x]),
            np.nanmin([np.nanmin(df[c].values) for c in self.y]),
            np.nanmax([np.nanmax(df[c].values) for c in self.y]),
            np.nanmin([np.nanmin(df[c].values) for c in self.y_stack]),
            np.nanmax([np.nanmax(df[c].values) for c in self.y_stack]),
        ]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, [2, 4]]), np.nanmax(r[:, [3, 5]])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_trapezoid_y = _build_draw_trapezoid_y(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_area = _build_extend_area_to_line_axis1_none_constant(
            draw_trapezoid_y, map_onto_pixel)
        x_names = self.x
        y_names = self.y
        y_stack_names = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = tuple(df[x_name].values for x_name in x_names)
            ys = tuple(df[y_name].values for y_name in y_names)
            y_stacks = tuple(df[y_stack_name].values
                             for y_stack_name in y_stack_names)

            cols = aggs + info(df)
            extend_area(vt, bounds, xs, ys, y_stacks, plot_start, *cols)

        return extend


class AreaToZeroAxis1XConstant(AreaToZeroAxis1):
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
        draw_trapezoid_y = _build_draw_trapezoid_y(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_area = _build_extend_area_to_zero_axis1_x_constant(
            draw_trapezoid_y, map_onto_pixel)

        x_values = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            ys = tuple(df[y_name].values for y_name in y_names)

            cols = aggs + info(df)
            # line may be clipped, then mapped to pixels
            extend_area(vt, bounds, x_values, ys, plot_start, *cols)

        return extend


class AreaToLineAxis1XConstant(AreaToLineAxis1):
    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(ycol)])
                    for ycol in self.y]):
            raise ValueError('y columns must be real')

        if not all([isreal(in_dshape.measure[str(ycol)])
                    for ycol in self.y_stack]):
            raise ValueError('y_stack columns must be real')

        unique_y_measures = set(in_dshape.measure[str(ycol)]
                                for ycol in self.y)
        if len(unique_y_measures) > 1:
            raise ValueError('y columns must have the same data type')

        unique_y_stack_measures = set(in_dshape.measure[str(ycol)]
                                      for ycol in self.y)
        if len(unique_y_stack_measures) > 1:
            raise ValueError('y_stack columns must have the same data type')

    def required_columns(self):
        return self.y + self.y_stack

    def compute_x_bounds(self, *args):
        x_min = np.nanmin(self.x)
        x_max = np.nanmax(self.x)
        return self.maybe_expand_bounds((x_min, x_max))

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin([np.nanmin(df[c].values) for c in self.y]),
            np.nanmax([np.nanmax(df[c].values) for c in self.y]),
            np.nanmin([np.nanmin(df[c].values) for c in self.y_stack]),
            np.nanmax([np.nanmax(df[c].values) for c in self.y_stack]),
        ]]
        )).compute()

        y_extents = np.nanmin(r[:, [0, 2]]), np.nanmax(r[:, [1, 3]])

        return (self.compute_x_bounds(),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_trapezoid_y = _build_draw_trapezoid_y(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_area = _build_extend_area_to_line_axis1_x_constant(
            draw_trapezoid_y, map_onto_pixel)

        x_values = self.x
        y_names = self.y
        y_stack_names = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            ys = tuple(df[y_name].values for y_name in y_names)
            y_stacks = tuple(df[y_stack_name].values
                             for y_stack_name in y_stack_names)

            cols = aggs + info(df)
            extend_area(vt, bounds, x_values, ys, y_stacks, plot_start, *cols)

        return extend


class AreaToZeroAxis1YConstant(AreaToZeroAxis1):
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
        draw_trapezoid_y = _build_draw_trapezoid_y(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_area = _build_extend_area_to_zero_axis1_y_constant(
            draw_trapezoid_y, map_onto_pixel)

        x_names = self.x
        y_values = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = tuple(df[x_name].values for x_name in x_names)

            cols = aggs + info(df)
            # line may be clipped, then mapped to pixels
            extend_area(vt, bounds, xs, y_values, plot_start, *cols)

        return extend


class AreaToLineAxis1YConstant(AreaToLineAxis1):
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
        y_min = min(np.nanmin(self.y), np.nanmin(self.y_stack))
        y_max = max(np.nanmax(self.y), np.nanmax(self.y_stack))
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
        draw_trapezoid_y = _build_draw_trapezoid_y(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_area = _build_extend_area_to_line_axis1_y_constant(
            draw_trapezoid_y, map_onto_pixel)
        x_names = self.x
        y_values = self.y
        y_stack_values = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = tuple(df[x_name].values for x_name in x_names)

            cols = aggs + info(df)
            extend_area(
                vt, bounds, xs, y_values, y_stack_values, plot_start, *cols)

        return extend


class AreaToZeroAxis1Ragged(_PointLike):
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
        return self.x, self.y

    def compute_x_bounds(self, df):
        bounds = self._compute_x_bounds(df[self.x].array.flat_array)
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds = self._compute_y_bounds(df[self.y].array.flat_array)

        # Make sure bounds include zero
        bounds = min(bounds[0], 0), max(bounds[1], 0)

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

        # Make sure y_extents include 0
        y_extents = min(y_extents[0], 0), max(y_extents[1], 0)

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_trapezoid_y = _build_draw_trapezoid_y(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_area = _build_extend_area_to_zero_axis1_ragged(
            draw_trapezoid_y, map_onto_pixel)
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = df[x_name].values
            ys = df[y_name].values
            cols = aggs + info(df)
            extend_area(vt, bounds, xs, ys, plot_start, *cols)

        return extend


class AreaToLineAxis1Ragged(_AreaToLineLike):
    def validate(self, in_dshape):
        try:
            from datashader.datatypes import RaggedDtype
        except ImportError:
            RaggedDtype = type(None)

        if not isinstance(in_dshape[str(self.x)], RaggedDtype):
            raise ValueError('x must be a RaggedArray')
        elif not isinstance(in_dshape[str(self.y)], RaggedDtype):
            raise ValueError('y must be a RaggedArray')
        elif not isinstance(in_dshape[str(self.y_stack)], RaggedDtype):
            raise ValueError('y_stack must be a RaggedArray')

    def required_columns(self):
        return self.x, self.y, self.y_stack

    def compute_x_bounds(self, df):
        bounds = self._compute_x_bounds(df[self.x].array.flat_array)
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds_y = self._compute_y_bounds(df[self.y].array.flat_array)
        bounds_y_stack = self._compute_y_bounds(
            df[self.y_stack].array.flat_array)

        # Make sure bounds include zero
        bounds = (min(bounds_y[0], bounds_y_stack[0], 0),
                  max(bounds_y[1], bounds_y_stack[1], 0))

        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin(df[self.x].array.flat_array),
            np.nanmax(df[self.x].array.flat_array),
            np.nanmin(df[self.y].array.flat_array),
            np.nanmax(df[self.y].array.flat_array),
            np.nanmin(df[self.y_stack].array.flat_array),
            np.nanmax(df[self.y_stack].array.flat_array),
        ]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, [2, 4]]), np.nanmax(r[:, [3, 5]])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_trapezoid_y = _build_draw_trapezoid_y(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        extend_area = _build_extend_area_to_line_axis1_ragged(
            draw_trapezoid_y, map_onto_pixel)
        x_name = self.x
        y_name = self.y
        y_stack_name = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            xs = df[x_name].values
            ys = df[y_name].values
            y_stacks = df[y_stack_name].values

            cols = aggs + info(df)
            extend_area(vt, bounds, xs, ys, y_stacks, plot_start, *cols)

        return extend


class Triangles(_PolygonLike):
    """An unstructured mesh of triangles, with vertices defined by ``xs`` and ``ys``.

    Parameters
    ----------
    xs, ys, zs : list of str
        Column names of x, y, and (optional) z coordinates of each vertex.
    """
    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        draw_triangle, draw_triangle_interp = _build_draw_triangle(append)
        map_onto_pixel = _build_map_onto_pixel_for_triangle(x_mapper, y_mapper)
        extend_triangles = _build_extend_triangles(draw_triangle, draw_triangle_interp, map_onto_pixel)
        weight_type = self.weight_type
        interpolate = self.interpolate

        def extend(aggs, df, vt, bounds, plot_start=True):
            cols = info(df)
            assert cols, 'There must be at least one column on which to aggregate'
            # mapped to pixels, then may be clipped
            extend_triangles(vt, bounds, df.values, weight_type, interpolate, aggs, cols)


        return extend


# -- Helpers for computing geometries --


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


def _build_map_onto_pixel_for_triangle(x_mapper, y_mapper):
    @ngjit
    def map_onto_pixel(vt, bounds, x, y):
        """Map points onto pixel grid.

        Points falling on upper bound are mapped into previous bin.
        """
        sx, tx, sy, ty = vt
        xmax, ymax = bounds[1], bounds[3]
        xx = int(x_mapper(x) * sx + tx)
        yy = int(y_mapper(y) * sy + ty)
        return (xx - 1 if x == xmax else xx,
                yy - 1 if y == ymax else yy)
    return map_onto_pixel


def _build_draw_line(append):
    """Specialize a line plotting kernel for a given append/axis combination"""
    @ngjit
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


def _build_draw_trapezoid_y(append):
    """Specialize a plotting kernel for drawing a trapezoid with two
    sides parallel to the y-axis"""

    @ngjit
    def clamp_y_indices(ystarti, ystopi, ymaxi):
        """Utility function to compute clamped y-indices"""

        # First, check if the y indices are both out of bounds in the same
        # direction. If this is true, then there is nothing to fill.
        #
        # Note that if the indices are out of bounds in opposite directions,
        # then the pair is not considered out of bounds because part of the
        # filled area between the coordinates will be in bounds.
        out_of_bounds = ((ystarti < 0 and ystopi <= 0) or
                         (ystarti > ymaxi and ystopi >= ymaxi))

        # Clamp ystarti to be within 0 and ymaxi
        clamped_ystarti = max(0, min(ymaxi, ystarti))

        # Following the Python range convention, clamp ystopi to be within
        # -1 and ymaxi+1.
        clamped_ystopi = max(-1, min(ymaxi + 1, ystopi))

        return out_of_bounds, clamped_ystarti, clamped_ystopi

    @ngjit
    def draw_trapezoid_y(x0i, x1i, y0i, y1i, y2i, y3i, xmaxi, ymaxi,
                         i, plot_start, clipped, stacked, *aggs_and_cols):
        """Draw a filled trapezoid that has two sides parallel to the y-axis

        Such a trapezoid is defined by two x coordinates (x0 for the left
        edge and x1 for the right parallel edge) and four y-coordinates
        (y0 for top left vertex, y1 for bottom left vertex, y2 for the bottom
        right vertex and y3 for the top right vertex).

                                          (x1, y3)
                                      _ +
                        (x0, y0)  _--   |
                                +       |
                                |       |
                                |       |
                                +       |
                        (x0, y1)  -     |
                                    -   |
                                      - |
                                        +
                                          (x1, y2)

        In a proper trapezoid (as drawn above), y0 >= y1 and y3 >= y2 so that
        edges do not intersect. This function also handles the case where
        y1 < y0 or y2 < y3, which results in a crossing edge.

        The trapezoid is filled using a vertical scan line algorithm where the
        start and stop bins are calculated by what amounts to a pair of
        Bresenham's line algorithms, one for the top edge and one for the
        bottom edge.

        Bins in the line connecting (x0, y1) and (x1, y2) are not filled if
        the `stacked` argument is set to True. This way stacked trapezoids
        will not have any overlapping bins.

        Parameters
        ----------
        x0i, x1i: int
            x-coordinate indices of the start and stop edge of the trapezoid
        y0i, y1i, y2i, y3i: int
            y-coordinate indices of the four corners of the trapezoid
        xmaxi, ymaxi: int
            The maximum allowable x-index and y-index respectively.
            The trapezoid will be clipped to these index values.
        i: int
            Group index
        plot_start: bool
            If True, the filled trapezoid will include the (x0, y0) to (x0, y1)
            edge. Otherwise this edge will not be included.
        clipped: bool
            If True, the filled trapezoid will include the (x1, y3) to (x1, y2)
            edge. Otherwise this edge will not be included.
        stacked: bool
            If False, the filled trapezoid will include the
            (x0, y1) to (x1, y2) edge. Otherwise this edge will not
            be included.
        """
        # Compute x-delta and which direction we need to iterate through
        # the x-bins
        dx = x1i - x0i
        ix = (dx > 0) - (dx < 0)

        # Compute y-delta and iteration direction for the top and bottom edge.
        # Note that these are numbered to match the y-coordinate of the start
        # of the line
        dy0 = y3i - y0i
        iy0 = (dy0 > 0) - (dy0 < 0)
        dy1 = y2i - y1i
        iy1 = (dy1 > 0) - (dy1 < 0)

        # Handle drawing the initial vertical line if plot_start is True
        if plot_start:
            y_oob, y_start, y_stop = clamp_y_indices(y0i, y1i, ymaxi)
            x_oob = x0i < 0 or x0i > xmaxi
            if y_oob or x_oob:
                # Out of bounds, nothing to do
                pass
            elif y_start == y_stop and not stacked:
                # No height, append to single bin if not in stacked mode
                append(i, x0i, y_start, *aggs_and_cols)
            else:
                # Non-zero height
                y = y_start
                iy = (y_start < y_stop) - (y_stop < y_start)

                if not stacked and -1 <= y_stop + iy <= ymaxi + 1:
                    # If not stacking, include bin on line from
                    # (x0, y1) to (x1, y2), otherwise leave it out
                    y_stop += iy

                while y != y_stop:
                    append(i, x0i, y, *aggs_and_cols)
                    y += iy

        # Handle zero width cases
        if dx == 0 and not clipped:
            y_oob, y_start, y_stop = clamp_y_indices(y3i, y2i, ymaxi)
            x_oob = x1i < 0 or x1i > xmaxi

            # y0-y1 edge already aggregated above if plot_start.
            # Now aggregate the y3-y2 edge
            if y_oob or x_oob:
                # Out of bounds, nothing to do
                pass
            elif y_start == y_stop and not stacked:
                # No height, append to single bin if not in stacked mode
                append(i, x1i, y_start, *aggs_and_cols)
            else:
                # Non-zero height
                y = y_start
                iy = (y_start < y_stop) - (y_stop < y_start)

                if not stacked and -1 <= y_stop + iy <= ymaxi + 1:
                    # If not stacking, include bin on line from
                    # (x0, y1) to (x1, y2), otherwise leave it
                    y_stop += iy

                while y != y_stop:
                    append(i, x1i, y, *aggs_and_cols)
                    y += iy
            return

        # Non-zero width.
        # Compute initial Bresenham line errors using the integer formulation.
        # Note that unlike that standard Bresenham's algorithm,
        # we are forcing the "driving axis" to be x regardless of the
        # relationship between x and y deltas.
        dx = abs(dx) * 2
        dy0 = abs(dy0) * 2
        dy1 = abs(dy1) * 2

        error0 = 2 * dy0 - dx
        error1 = 2 * dy1 - dx

        while x0i != x1i:
            # Update error and y-bin index for y0 line.
            #
            # Note that in the standard Bresenham's algorithm this would be
            # an if statement.  We need to make this a while statement in our
            # case because we are forcing the x-axis to be the driving axis
            # which requires the ability to increment the y-index multiple
            # times for each step in the x-index.
            while error0 >= 0 and (error0 or ix > 0):
                error0 -= 2 * dx
                y0i += iy0

            error0 += 2 * dy0

            # Update error and y-bin index for y1 line
            while error1 >= 0 and (error1 or ix > 0):
                error1 -= 2 * dx
                y1i += iy1

            error1 += 2 * dy1

            # Update x
            x0i += ix

            # Check if x0i is in bounds.
            x_oob = x0i < 0 or x0i > xmaxi
            if x_oob:
                # Note that it's important that we have already updated the
                # error values and y indices before continuing to the next
                # loop iteration
                continue

            # Compute clamped y indices
            y_oob, y_start, y_stop = clamp_y_indices(y0i, y1i, ymaxi)

            if y_oob:
                # Out of bounds, nothing to do
                pass
            elif y_start == y_stop and not stacked:
                # No height case, append to single bin if not in stacked mode
                append(i, x0i, y_start, *aggs_and_cols)
            else:
                # Non-zero height case,
                # append to bins in trapezoid column
                y = y_start
                iy = (y_start < y_stop) - (y_stop < y_start)

                if not stacked and -1 <= y_stop + iy <= ymaxi + 1:
                    # If not stacking, aggregate bin on line from
                    # (x0, y1) to (x1, y2), otherwise leave it out
                    y_stop += iy

                while y != y_stop:
                    append(i, x0i, y, *aggs_and_cols)
                    y += iy

    return draw_trapezoid_y


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


@ngjit
def _skip_or_clip_trapezoid_y(x0, x1, y0, y1, y2, y3, bounds, plot_start):
    xmin, xmax, ymin, ymax = bounds
    skip = False
    clipped = False

    # If any of the coordinates are NaN, there's a discontinuity.
    # Skip the entire trapezoid.
    if (np.isnan(x0) or
            np.isnan(x1) or
            np.isnan(y0) or
            np.isnan(y1) or
            np.isnan(y2) or
            np.isnan(y3)):
        plot_start = True
        skip = True

    # Check if trapezoid is out of bounds vertically
    if ((y0 > ymax and y1 > ymax and y2 > ymax and y3 > ymax) or
            (y0 < ymin and y1 < ymin and y2 < ymin and y3 < ymin)):
        # No need to perform clipping, we will skip the whole thing
        plot_start = True
        skip = True
        return x0, x1, y0, y1, y2, y3, skip, clipped, plot_start

    # Initialize Liang-Barsky parameters
    t0, t1 = 0, 1
    dx = x1 - x0
    dy0 = y3 - y0
    dy1 = y2 - y1

    # Clip on x boundaries only
    t0, t1, accept = _clipt(-dx, x0 - xmin, t0, t1)
    if not accept:
        skip = True

    t0, t1, accept = _clipt(dx, xmax - x0, t0, t1)
    if not accept:
        skip = True

    # Update lines on clipping
    if t1 < 1:
        clipped = True
        x1 = x0 + t1 * dx
        y2 = y1 + t1 * dy1
        y3 = y0 + t1 * dy0

    if t0 > 0:
        # If x0 is clipped, we need to plot the new start
        clipped = True
        plot_start = True
        x0 = x0 + t0 * dx
        y0 = y0 + t0 * dy0
        y1 = y1 + t0 * dy1

    return x0, x1, y0, y1, y2, y3, skip, clipped, plot_start


def _build_extend_line_axis0(draw_line, map_onto_pixel):
    @ngjit
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


def _build_extend_line_axis0_multi(draw_line, map_onto_pixel):
    @ngjit
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


def _build_extend_line_axis1_none_constant(draw_line, map_onto_pixel):
    @ngjit
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


def _build_extend_line_axis1_x_constant(draw_line, map_onto_pixel):
    @ngjit
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


def _build_extend_line_axis1_y_constant(draw_line, map_onto_pixel):
    @ngjit
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


def _build_extend_line_axis1_ragged(draw_line, map_onto_pixel):

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


def _build_extend_area_to_zero_axis0(draw_trapezoid_y, map_onto_pixel):
    @ngjit
    def extend_area(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""

        stacked = False
        xmaxi, ymaxi = map_onto_pixel(vt, bounds, bounds[1], bounds[3])

        nrows = xs.shape[0]
        i = 0
        while i < nrows - 1:
            x0 = xs[i]
            x1 = xs[i + 1]

            y0 = ys[i]
            y1 = 0.0
            y2 = 0.0
            y3 = ys[i + 1]

            x0, x1, y0, y1, y2, y3, skip, clipped, plot_start = \
                _skip_or_clip_trapezoid_y(
                    x0, x1, y0, y1, y2, y3, bounds, plot_start)

            if not skip:
                x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                _, y1i = map_onto_pixel(vt, bounds, x0, y1)
                y2i = y1i
                x1i, y3i = map_onto_pixel(vt, bounds, x1, y3)

                draw_trapezoid_y(x0i, x1i, y0i, y1i, y2i, y3i, xmaxi, ymaxi,
                                 i, plot_start, clipped, stacked,
                                 *aggs_and_cols)
                plot_start = False
            i += 1

    return extend_area


def _build_extend_area_to_line_axis0(draw_trapezoid_y, map_onto_pixel):
    @ngjit
    def extend_area(vt, bounds, xs, ys0, ys1, plot_start, *aggs_and_cols):
        """Aggregate filled area between the line formed by
        ``xs`` and ``ys0`` and the line formed by ``xs`` and ``ys1``"""

        stacked = True
        xmaxi, ymaxi = map_onto_pixel(vt, bounds, bounds[1], bounds[3])

        nrows = xs.shape[0]
        i = 0
        while i < nrows - 1:
            x0 = xs[i]
            x1 = xs[i + 1]

            y0 = ys0[i]
            y1 = ys1[i]
            y2 = ys1[i + 1]
            y3 = ys0[i + 1]

            x0, x1, y0, y1, y2, y3, skip, clipped, plot_start = \
                _skip_or_clip_trapezoid_y(
                    x0, x1, y0, y1, y2, y3, bounds, plot_start)

            if not skip:
                x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                _, y1i = map_onto_pixel(vt, bounds, x0, y1)
                _, y2i = map_onto_pixel(vt, bounds, x1, y2)
                x1i, y3i = map_onto_pixel(vt, bounds, x1, y3)

                draw_trapezoid_y(x0i, x1i, y0i, y1i, y2i, y3i, xmaxi, ymaxi,
                                 i, plot_start, clipped, stacked,
                                 *aggs_and_cols)
                plot_start = False
            i += 1

    return extend_area


def _build_extend_area_to_zero_axis0_multi(draw_trapezoid_y, map_onto_pixel):
    @ngjit
    def extend_area(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""

        stacked = False
        xmaxi, ymaxi = map_onto_pixel(vt, bounds, bounds[1], bounds[3])

        nrows = xs[0].shape[0]
        ncols = len(xs)

        orig_plot_start = plot_start

        j = 0
        while j < ncols:
            plot_start = orig_plot_start
            i = 0
            while i < nrows - 1:
                x0 = xs[j][i]
                x1 = xs[j][i + 1]

                y0 = ys[j][i]
                y1 = 0.0
                y2 = 0.0
                y3 = ys[j][i + 1]

                x0, x1, y0, y1, y2, y3, skip, clipped, plot_start = \
                    _skip_or_clip_trapezoid_y(
                        x0, x1, y0, y1, y2, y3, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    _, y1i = map_onto_pixel(vt, bounds, x0, y1)
                    y2i = y1i
                    x1i, y3i = map_onto_pixel(vt, bounds, x1, y3)

                    draw_trapezoid_y(x0i, x1i, y0i, y1i, y2i, y3i, xmaxi, ymaxi,
                                     i, plot_start, clipped, stacked,
                                     *aggs_and_cols)
                    plot_start = False
                i += 1
            j += 1

    return extend_area


def _build_extend_area_to_line_axis0_multi(draw_trapezoid_y, map_onto_pixel):
    @ngjit
    def extend_area(vt, bounds, xs, ys0, ys1, plot_start, *aggs_and_cols):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""

        stacked = True
        xmaxi, ymaxi = map_onto_pixel(vt, bounds, bounds[1], bounds[3])

        nrows = xs[0].shape[0]
        ncols = len(xs)
        orig_plot_start = plot_start

        j = 0
        while j < ncols:
            plot_start = orig_plot_start
            i = 0
            while i < nrows - 1:
                x0 = xs[j][i]
                x1 = xs[j][i + 1]

                y0 = ys0[j][i]
                y1 = ys1[j][i]
                y2 = ys1[j][i + 1]
                y3 = ys0[j][i + 1]

                x0, x1, y0, y1, y2, y3, skip, clipped, plot_start = \
                    _skip_or_clip_trapezoid_y(
                        x0, x1, y0, y1, y2, y3, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    _, y1i = map_onto_pixel(vt, bounds, x0, y1)
                    _, y2i = map_onto_pixel(vt, bounds, x1, y2)
                    x1i, y3i = map_onto_pixel(vt, bounds, x1, y3)

                    draw_trapezoid_y(x0i, x1i, y0i, y1i, y2i, y3i, xmaxi,
                                     ymaxi,
                                     i, plot_start, clipped, stacked,
                                     *aggs_and_cols)
                    plot_start = False
                i += 1
            j += 1

    return extend_area


def _build_extend_area_to_zero_axis1_none_constant(draw_trapezoid_y, map_onto_pixel):
    @ngjit
    def extend_area(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""

        stacked = False
        xmaxi, ymaxi = map_onto_pixel(vt, bounds, bounds[1], bounds[3])

        nrows = xs[0].shape[0]
        ncols = len(xs)

        i = 0
        while i < nrows:
            plot_start = True
            j = 0
            while j < ncols - 1:
                x0 = xs[j][i]
                x1 = xs[j + 1][i]

                y0 = ys[j][i]
                y1 = 0.0
                y2 = 0.0
                y3 = ys[j + 1][i]

                x0, x1, y0, y1, y2, y3, skip, clipped, plot_start = \
                    _skip_or_clip_trapezoid_y(
                        x0, x1, y0, y1, y2, y3, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    _, y1i = map_onto_pixel(vt, bounds, x0, y1)
                    y2i = y1i
                    x1i, y3i = map_onto_pixel(vt, bounds, x1, y3)

                    draw_trapezoid_y(x0i, x1i, y0i, y1i, y2i, y3i, xmaxi, ymaxi,
                                     i, plot_start, clipped, stacked,
                                     *aggs_and_cols)
                    plot_start = False
                j += 1
            i += 1

    return extend_area


def _build_extend_area_to_line_axis1_none_constant(draw_trapezoid_y, map_onto_pixel):
    @ngjit
    def extend_area(vt, bounds, xs, ys0, ys1, plot_start, *aggs_and_cols):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""

        stacked = True
        xmaxi, ymaxi = map_onto_pixel(vt, bounds, bounds[1], bounds[3])

        nrows = xs[0].shape[0]
        ncols = len(xs)

        i = 0
        while i < nrows:
            plot_start = True
            j = 0
            while j < ncols - 1:
                x0 = xs[j][i]
                x1 = xs[j + 1][i]

                y0 = ys0[j][i]
                y1 = ys1[j][i]
                y2 = ys1[j + 1][i]
                y3 = ys0[j + 1][i]

                x0, x1, y0, y1, y2, y3, skip, clipped, plot_start = \
                    _skip_or_clip_trapezoid_y(
                        x0, x1, y0, y1, y2, y3, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    _, y1i = map_onto_pixel(vt, bounds, x0, y1)
                    _, y2i = map_onto_pixel(vt, bounds, x1, y2)
                    x1i, y3i = map_onto_pixel(vt, bounds, x1, y3)

                    draw_trapezoid_y(x0i, x1i, y0i, y1i, y2i, y3i, xmaxi, ymaxi,
                                     i, plot_start, clipped, stacked,
                                     *aggs_and_cols)
                    plot_start = False
                j += 1
            i += 1

    return extend_area


def _build_extend_area_to_zero_axis1_x_constant(draw_trapezoid_y, map_onto_pixel):
    @ngjit
    def extend_area(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""

        stacked = False
        xmaxi, ymaxi = map_onto_pixel(vt, bounds, bounds[1], bounds[3])

        nrows = ys[0].shape[0]
        ncols = len(ys)

        i = 0
        while i < nrows:
            plot_start = True
            j = 0
            while j < ncols - 1:
                x0 = xs[j]
                x1 = xs[j + 1]

                y0 = ys[j][i]
                y1 = 0.0
                y2 = 0.0
                y3 = ys[j + 1][i]

                x0, x1, y0, y1, y2, y3, skip, clipped, plot_start = \
                    _skip_or_clip_trapezoid_y(
                        x0, x1, y0, y1, y2, y3, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    _, y1i = map_onto_pixel(vt, bounds, x0, y1)
                    y2i = y1i
                    x1i, y3i = map_onto_pixel(vt, bounds, x1, y3)

                    draw_trapezoid_y(x0i, x1i, y0i, y1i, y2i, y3i, xmaxi, ymaxi,
                                     i, plot_start, clipped, stacked,
                                     *aggs_and_cols)
                    plot_start = False
                j += 1
            i += 1

    return extend_area


def _build_extend_area_to_line_axis1_x_constant(draw_trapezoid_y, map_onto_pixel):
    @ngjit
    def extend_area(vt, bounds, xs, ys0, ys1, plot_start, *aggs_and_cols):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""

        stacked = False
        xmaxi, ymaxi = map_onto_pixel(vt, bounds, bounds[1], bounds[3])

        nrows = ys0[0].shape[0]
        ncols = len(ys0)

        i = 0
        while i < nrows:
            plot_start = True
            j = 0
            while j < ncols - 1:
                x0 = xs[j]
                x1 = xs[j + 1]

                y0 = ys0[j][i]
                y1 = ys1[j][i]
                y2 = ys1[j + 1][i]
                y3 = ys0[j + 1][i]

                x0, x1, y0, y1, y2, y3, skip, clipped, plot_start = \
                    _skip_or_clip_trapezoid_y(
                        x0, x1, y0, y1, y2, y3, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    _, y1i = map_onto_pixel(vt, bounds, x0, y1)
                    _, y2i = map_onto_pixel(vt, bounds, x1, y2)
                    x1i, y3i = map_onto_pixel(vt, bounds, x1, y3)

                    draw_trapezoid_y(x0i, x1i, y0i, y1i, y2i, y3i, xmaxi, ymaxi,
                                     i, plot_start, clipped, stacked,
                                     *aggs_and_cols)
                    plot_start = False
                j += 1
            i += 1

    return extend_area


def _build_extend_area_to_zero_axis1_y_constant(draw_trapezoid_y, map_onto_pixel):
    @ngjit
    def extend_area(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""

        stacked = False
        xmaxi, ymaxi = map_onto_pixel(vt, bounds, bounds[1], bounds[3])

        nrows = xs[0].shape[0]
        ncols = len(xs)

        i = 0
        while i < nrows:
            plot_start = True
            j = 0
            while j < ncols - 1:
                x0 = xs[j][i]
                x1 = xs[j + 1][i]

                y0 = ys[j]
                y1 = 0.0
                y2 = 0.0
                y3 = ys[j + 1]

                x0, x1, y0, y1, y2, y3, skip, clipped, plot_start = \
                    _skip_or_clip_trapezoid_y(
                        x0, x1, y0, y1, y2, y3, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    _, y1i = map_onto_pixel(vt, bounds, x0, y1)
                    y2i = y1i
                    x1i, y3i = map_onto_pixel(vt, bounds, x1, y3)

                    draw_trapezoid_y(x0i, x1i, y0i, y1i, y2i, y3i, xmaxi, ymaxi,
                                     i, plot_start, clipped, stacked,
                                     *aggs_and_cols)
                    plot_start = False
                j += 1
            i += 1

    return extend_area


def _build_extend_area_to_line_axis1_y_constant(draw_trapezoid_y, map_onto_pixel):
    @ngjit
    def extend_area(vt, bounds, xs, ys0, ys1, plot_start, *aggs_and_cols):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""

        stacked = True
        xmaxi, ymaxi = map_onto_pixel(vt, bounds, bounds[1], bounds[3])

        nrows = xs[0].shape[0]
        ncols = len(xs)

        i = 0
        while i < nrows:
            plot_start = True
            j = 0
            while j < ncols - 1:
                x0 = xs[j][i]
                x1 = xs[j + 1][i]

                y0 = ys0[j]
                y1 = ys1[j]
                y2 = ys1[j + 1]
                y3 = ys0[j + 1]

                x0, x1, y0, y1, y2, y3, skip, clipped, plot_start = \
                    _skip_or_clip_trapezoid_y(
                        x0, x1, y0, y1, y2, y3, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    _, y1i = map_onto_pixel(vt, bounds, x0, y1)
                    _, y2i = map_onto_pixel(vt, bounds, x1, y2)
                    x1i, y3i = map_onto_pixel(vt, bounds, x1, y3)

                    draw_trapezoid_y(x0i, x1i, y0i, y1i, y2i, y3i, xmaxi, ymaxi,
                                     i, plot_start, clipped, stacked,
                                     *aggs_and_cols)
                    plot_start = False
                j += 1
            i += 1

    return extend_area


def _build_extend_area_to_zero_axis1_ragged(draw_trapezoid_y, map_onto_pixel):

    def extend_line(vt, bounds, xs, ys, plot_start, *aggs_and_cols):
        x_start_indices = xs.start_indices
        x_flat_array = xs.flat_array

        y_start_indices = ys.start_indices
        y_flat_array = ys.flat_array

        perform_extend_area_to_zero_axis1_ragged(
            vt, bounds, x_start_indices, x_flat_array,
            y_start_indices, y_flat_array, plot_start, *aggs_and_cols)

    @ngjit
    def perform_extend_area_to_zero_axis1_ragged(
            vt, bounds, x_start_indices, x_flat_array,
            y_start_indices, y_flat_array, plot_start, *aggs_and_cols):

        stacked = False
        xmaxi, ymaxi = map_onto_pixel(vt, bounds, bounds[1], bounds[3])

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
                x1 = x_flat_array[x_start_index + j + 1]

                y0 = y_flat_array[y_start_index + j]
                y1 = 0.0
                y2 = 0.0
                y3 = y_flat_array[y_start_index + j + 1]

                x0, x1, y0, y1, y2, y3, skip, clipped, plot_start = \
                    _skip_or_clip_trapezoid_y(
                        x0, x1, y0, y1, y2, y3, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    _, y1i = map_onto_pixel(vt, bounds, x0, y1)
                    y2i = y1i
                    x1i, y3i = map_onto_pixel(vt, bounds, x1, y3)

                    draw_trapezoid_y(
                        x0i, x1i, y0i, y1i, y2i, y3i, xmaxi, ymaxi,
                        i, plot_start, clipped, stacked, *aggs_and_cols)

                    plot_start = False

                j += 1
            i += 1

    return extend_line


def _build_extend_area_to_line_axis1_ragged(draw_trapezoid_y, map_onto_pixel):

    def extend_line(vt, bounds, xs, ys0, ys1, plot_start, *aggs_and_cols):
        x_start_indices = xs.start_indices
        x_flat_array = xs.flat_array

        y0_start_indices = ys0.start_indices
        y0_flat_array = ys0.flat_array

        y1_start_indices = ys1.start_indices
        y1_flat_array = ys1.flat_array

        perform_extend_area_to_line_axis1_ragged(
            vt, bounds, x_start_indices, x_flat_array,
            y0_start_indices, y0_flat_array, y1_start_indices, y1_flat_array,
            plot_start, *aggs_and_cols)

    @ngjit
    def perform_extend_area_to_line_axis1_ragged(
            vt, bounds, x_start_indices, x_flat_array,
            y0_start_indices, y0_flat_array, y1_start_indices, y1_flat_array,
            plot_start, *aggs_and_cols):

        stacked = True
        xmaxi, ymaxi = map_onto_pixel(vt, bounds, bounds[1], bounds[3])

        nrows = len(x_start_indices)
        x_flat_len = len(x_flat_array)
        y0_flat_len = len(y0_flat_array)
        y1_flat_len = len(y1_flat_array)

        i = 0
        while i < nrows:
            plot_start = True

            # Get x index range
            x_start_index = x_start_indices[i]
            x_stop_index = (x_start_indices[i + 1]
                            if i < nrows - 1
                            else x_flat_len)

            # Get y index range
            y0_start_index = y0_start_indices[i]
            y0_stop_index = (y0_start_indices[i + 1]
                             if i < nrows - 1
                             else y0_flat_len)

            y1_start_index = y1_start_indices[i]
            y1_stop_index = (y1_start_indices[i + 1]
                             if i < nrows - 1
                             else y1_flat_len)

            # Find line segment length as shorter of the two segments
            segment_len = min(x_stop_index - x_start_index,
                              y0_stop_index - y0_start_index,
                              y1_stop_index - y1_start_index)

            j = 0
            while j < segment_len - 1:

                x0 = x_flat_array[x_start_index + j]
                x1 = x_flat_array[x_start_index + j + 1]

                y0 = y0_flat_array[y0_start_index + j]
                y1 = y1_flat_array[y1_start_index + j]
                y2 = y1_flat_array[y1_start_index + j + 1]
                y3 = y0_flat_array[y0_start_index + j + 1]

                x0, x1, y0, y1, y2, y3, skip, clipped, plot_start = \
                    _skip_or_clip_trapezoid_y(
                        x0, x1, y0, y1, y2, y3, bounds, plot_start)

                if not skip:
                    x0i, y0i = map_onto_pixel(vt, bounds, x0, y0)
                    _, y1i = map_onto_pixel(vt, bounds, x0, y1)
                    _, y2i = map_onto_pixel(vt, bounds, x1, y2)
                    x1i, y3i = map_onto_pixel(vt, bounds, x1, y3)

                    draw_trapezoid_y(
                        x0i, x1i, y0i, y1i, y2i, y3i, xmaxi, ymaxi,
                        i, plot_start, clipped, stacked, *aggs_and_cols)

                    plot_start = False

                j += 1
            i += 1

    return extend_line


def _build_draw_triangle(append):
    """Specialize a triangle plotting kernel for a given append/axis combination"""
    @ngjit
    def edge_func(ax, ay, bx, by, cx, cy):
        return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax)

    @ngjit
    def draw_triangle_interp(verts, bbox, biases, aggs, weights):
        """Same as `draw_triangle()`, but with weights interpolated from vertex
        values.
        """
        minx, maxx, miny, maxy = bbox
        w0, w1, w2 = weights
        if minx == maxx and miny == maxy:
            # Subpixel case; area == 0
            append(minx, miny, aggs, (w0 + w1 + w2) / 3)
        else:
            (ax, ay), (bx, by), (cx, cy) = verts
            bias0, bias1, bias2 = biases
            area = edge_func(ax, ay, bx, by, cx, cy)
            for j in range(miny, maxy+1):
                for i in range(minx, maxx+1):
                    g2 = edge_func(ax, ay, bx, by, i, j)
                    g0 = edge_func(bx, by, cx, cy, i, j)
                    g1 = edge_func(cx, cy, ax, ay, i, j)
                    if ((g2 + bias0) | (g0 + bias1) | (g1 + bias2)) >= 0:
                        interp_res = (g0 * w0 + g1 * w1 + g2 * w2) / area
                        append(i, j, aggs, interp_res)

    @ngjit
    def draw_triangle(verts, bbox, biases, aggs, val):
        """Draw a triangle on a grid.

        Plots a triangle with integer coordinates onto a pixel grid,
        clipping to the bounds. The vertices are assumed to have
        already been scaled and transformed.
        """
        minx, maxx, miny, maxy = bbox
        if minx == maxx and miny == maxy:
            # Subpixel case; area == 0
            append(minx, miny, aggs, val)
        else:
            (ax, ay), (bx, by), (cx, cy) = verts
            bias0, bias1, bias2 = biases
            for j in range(miny, maxy+1):
                for i in range(minx, maxx+1):
                    if ((edge_func(ax, ay, bx, by, i, j) + bias0) >= 0 and
                            (edge_func(bx, by, cx, cy, i, j) + bias1) >= 0 and
                            (edge_func(cx, cy, ax, ay, i, j) + bias2) >= 0):
                        append(i, j, aggs, val)


    return draw_triangle, draw_triangle_interp


def _build_extend_triangles(draw_triangle, draw_triangle_interp, map_onto_pixel):
    @ngjit
    def extend_triangles(vt, bounds, verts, weight_type, interpolate, aggs, cols):
        """Aggregate along an array of triangles formed by arrays of CW
        vertices. Each row corresponds to a single triangle definition.

        `weight_type == True` means "weights are on vertices"
        """
        xmin, xmax, ymin, ymax = bounds
        cmax_x, cmax_y = max(xmin, xmax), max(ymin, ymax)
        cmin_x, cmin_y = min(xmin, xmax), min(ymin, ymax)
        vmax_x, vmax_y = map_onto_pixel(vt, bounds, cmax_x, cmax_y)
        vmin_x, vmin_y = map_onto_pixel(vt, bounds, cmin_x, cmin_y)

        col = cols[0] # Only aggregate over one column, for now
        n_tris = verts.shape[0]
        for n in range(0, n_tris, 3):
            a = verts[n]
            b = verts[n+1]
            c = verts[n+2]
            axn, ayn = a[0], a[1]
            bxn, byn = b[0], b[1]
            cxn, cyn = c[0], c[1]
            col0, col1, col2 = col[n], col[n+1], col[n+2]

            # Map triangle vertices onto pixels
            ax, ay = map_onto_pixel(vt, bounds, axn, ayn)
            bx, by = map_onto_pixel(vt, bounds, bxn, byn)
            cx, cy = map_onto_pixel(vt, bounds, cxn, cyn)

            # Get bounding box
            minx = min(ax, bx, cx)
            maxx = max(ax, bx, cx)
            miny = min(ay, by, cy)
            maxy = max(ay, by, cy)

            # Skip any further processing of triangles outside of viewing area
            if (minx >= vmax_x or
                    maxx < vmin_x or
                    miny >= vmax_y or
                    maxy < vmin_y):
                continue

            # Clip bbox to viewing area
            minx = max(minx, vmin_x)
            maxx = min(maxx, vmax_x)
            miny = max(miny, vmin_y)
            maxy = min(maxy, vmax_y)

            # Prevent double-drawing edges.
            # https://msdn.microsoft.com/en-us/library/windows/desktop/bb147314(v=vs.85).aspx
            bias0, bias1, bias2 = -1, -1, -1
            if ay < by or (by == ay and ax < bx):
                 bias0 = 0
            if by < cy or (cy == by and bx < cx):
                 bias1 = 0
            if cy < ay or (ay == cy and cx < ax):
                 bias2 = 0

            bbox = minx, maxx, miny, maxy
            biases = bias0, bias1, bias2
            mapped_verts = (ax, ay), (bx, by), (cx, cy)

            # draw triangles (will be clipped where outside bounds)
            if interpolate:
                weights = col0, col1, col2
                draw_triangle_interp(mapped_verts, bbox, biases, aggs, weights)
            else:
                val = (col[n] + col[n+1] + col[n+2]) / 3
                draw_triangle(mapped_verts, bbox, biases, aggs, val)

    return extend_triangles
