from __future__ import annotations
import numpy as np
from toolz import memoize

from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda

try:
    import cudf
    import cupy as cp
    from ..transfer_functions._cuda_utils import cuda_args
except ImportError:
    cudf = None
    cp = None
    cuda_args = None


class _AreaToLineLike(Glyph):
    """Shared methods between Point and Line"""
    def __init__(self, x, y, y_stack):
        self.x = x
        self.y = y
        self.y_stack = y_stack

    @property
    def ndims(self):
        return 1

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
        bounds = self._compute_bounds(df[self.x])
        return self.maybe_expand_bounds(bounds)


class AreaToZeroAxis0(_PointLike):
    """A filled area glyph.
    The area to be filled is the region from the line defined by ``x`` and
    ``y`` and the y=0 line

    Parameters
    ----------
    x, y
        Column names for the x and y coordinates of each vertex.
    """

    def compute_y_bounds(self, df):
        # Compute bounds of curve
        bounds = self._compute_bounds(df[self.y])

        # Make sure bounds include zero
        bounds = min(bounds[0], 0), max(bounds[1], 0)

        # Expand bounds if needed
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

        # Make sure y_extents include zero
        y_extents = min(y_extents[0], 0), max(y_extents[1], 0)

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(
            append, map_onto_pixel, expand_aggs_and_cols
        )

        extend_cpu, extend_cuda = _build_extend_area_to_zero_axis0(
            draw_trapezoid_y, expand_aggs_and_cols
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
                xs = df[x_name].values
                ys = df[y_name].values
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                plot_start, xs, ys, *aggs_and_cols
            )

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
        bounds0 = self._compute_bounds(df[self.y])
        bounds1 = self._compute_bounds(df[self.y_stack])

        # Combine bounds from the two curves
        bounds = min(bounds0[0], bounds1[0]), max(bounds0[1], bounds1[1])

        # Expand bounds if needed
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin(df[self.x].values).item(),
            np.nanmax(df[self.x].values).item(),
            np.nanmin(df[self.y].values).item(),
            np.nanmax(df[self.y].values).item(),
            np.nanmin(df[self.y_stack].values).item(),
            np.nanmax(df[self.y_stack].values).item()]]
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
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(
            append, map_onto_pixel, expand_aggs_and_cols
        )
        extend_cpu, extend_cuda = _build_extend_area_to_line_axis0(
            draw_trapezoid_y, expand_aggs_and_cols
        )
        x_name = self.x
        y_name = self.y
        y_stack_name = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, x_name)
                ys0 = self.to_cupy_array(df, y_name)
                ys1 = self.to_cupy_array(df, y_stack_name)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df[x_name].values
                ys0 = df[y_name].values
                ys1 = df[y_stack_name].values
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                plot_start, xs, ys0, ys1, *aggs_and_cols
            )

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
        bounds_list = [self._compute_bounds(df[x])
                       for x in self.x]
        mins, maxes = zip(*bounds_list)
        return self.maybe_expand_bounds((min(mins), max(maxes)))

    def compute_y_bounds(self, df):
        bounds_list = [self._compute_bounds(df[y])
                       for y in self.y]
        mins, maxes = zip(*bounds_list)

        # Make sure min/max range includes zero
        mn = min(0, min(mins))
        mx = max(0, max(maxes))
        return self.maybe_expand_bounds((mn, mx))

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

        # Makes sure y_extents include 0
        y_extents = min(0, y_extents[0]), max(0, y_extents[1])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(
            append, map_onto_pixel, expand_aggs_and_cols
        )

        extend_cpu, extend_cuda = _build_extend_area_to_zero_axis0_multi(
            draw_trapezoid_y, expand_aggs_and_cols
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
                sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                plot_start, xs, ys, *aggs_and_cols
            )

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
        bounds_list = [self._compute_bounds(df[x])
                       for x in self.x]
        mins, maxes = zip(*bounds_list)
        return self.maybe_expand_bounds((min(mins), max(maxes)))

    def compute_y_bounds(self, df):
        bounds_list = [self._compute_bounds(df[y])
                       for y in self.y + self.y_stack]
        mins, maxes = zip(*bounds_list)

        return self.maybe_expand_bounds((min(mins), max(maxes)))

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin([np.nanmin(df[c].values).item() for c in self.x]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.x]),
            np.nanmin([np.nanmin(df[c].values).item() for c in self.y]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.y]),
            np.nanmin([np.nanmin(df[c].values).item() for c in self.y_stack]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.y_stack]),
        ]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, [2, 4]]), np.nanmax(r[:, [3, 5]])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(
            append, map_onto_pixel, expand_aggs_and_cols
        )
        extend_cpu, extend_cuda = _build_extend_area_to_line_axis0_multi(
            draw_trapezoid_y, expand_aggs_and_cols
        )
        x_names = self.x
        y_names = self.y
        y_stack_names = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, x_names)
                ys0 = self.to_cupy_array(df, y_names)
                ys1 = self.to_cupy_array(df, y_stack_names)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys0 = df.loc[:, list(y_names)].to_numpy()
                ys1 = df.loc[:, list(y_stack_names)].to_numpy()
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                plot_start, xs, ys0, ys1, *aggs_and_cols
            )

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

        bounds_list = [self._compute_bounds(xcol) for xcol in xs]
        mins, maxes = zip(*bounds_list)

        return self.maybe_expand_bounds((min(mins), max(maxes)))

    def compute_y_bounds(self, df):
        ys = tuple(df[ylabel] for ylabel in self.y)

        bounds_list = [self._compute_bounds(ycol) for ycol in ys]
        mins, maxes = zip(*bounds_list)

        # Make sure min/max range includes zero
        mn = min(0, min(mins))
        mx = max(0, max(maxes))

        return self.maybe_expand_bounds((mn, mx))

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

        # Makes sure y_extents include 0
        y_extents = min(0, y_extents[0]), max(0, y_extents[1])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(
            append, map_onto_pixel, expand_aggs_and_cols
        )

        extend_cpu, extend_cuda = _build_extend_area_to_zero_axis1_none_constant(
            draw_trapezoid_y, expand_aggs_and_cols
        )

        x_names = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df,list(x_names))
                ys = self.to_cupy_array(df,list(y_names))
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys = df.loc[:, list(y_names)].to_numpy()
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
            )

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

        bounds_list = [self._compute_bounds(xcol) for xcol in xs]
        mins, maxes = zip(*bounds_list)

        return self.maybe_expand_bounds((min(mins), max(maxes)))

    def compute_y_bounds(self, df):
        bounds_list = [self._compute_bounds(df[y])
                       for y in self.y + self.y_stack]
        mins, maxes = zip(*bounds_list)

        return self.maybe_expand_bounds((min(mins), max(maxes)))

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin([np.nanmin(df[c].values).item() for c in self.x]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.x]),
            np.nanmin([np.nanmin(df[c].values).item() for c in self.y]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.y]),
            np.nanmin([np.nanmin(df[c].values).item() for c in self.y_stack]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.y_stack]),
        ]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, [2, 4]]), np.nanmax(r[:, [3, 5]])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(
            append, map_onto_pixel, expand_aggs_and_cols
        )
        extend_cpu, extend_cuda = _build_extend_area_to_line_axis1_none_constant(
            draw_trapezoid_y, expand_aggs_and_cols
        )
        x_names = self.x
        y_names = self.y
        y_stack_names = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df,list(x_names))
                ys = self.to_cupy_array(df,list(y_names))
                y_stacks = self.to_cupy_array(df,list(y_stack_names))
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys = df.loc[:, list(y_names)].to_numpy()
                y_stacks = df.loc[:, list(y_stack_names)].to_numpy()
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                xs, ys, y_stacks, *aggs_and_cols
            )

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
            np.nanmin([np.nanmin(df[c].values).item() for c in self.y]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.y])]]
        )).compute()

        y_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])

        return (self.compute_x_bounds(),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(
            append, map_onto_pixel, expand_aggs_and_cols
        )
        extend_cpu, extend_cuda = _build_extend_area_to_zero_axis1_x_constant(
            draw_trapezoid_y, expand_aggs_and_cols
        )

        x_values = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                ys = self.to_cupy_array(df,list(y_names))
                do_extend = extend_cuda[cuda_args(ys.shape)]
            else:
                ys = df.loc[:, list(y_names)].to_numpy()
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                x_values, ys, *aggs_and_cols
            )

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
            np.nanmin([np.nanmin(df[c].values).item() for c in self.y]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.y]),
            np.nanmin([np.nanmin(df[c].values).item() for c in self.y_stack]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.y_stack]),
        ]]
        )).compute()

        y_extents = np.nanmin(r[:, [0, 2]]), np.nanmax(r[:, [1, 3]])

        return (self.compute_x_bounds(),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(
            append, map_onto_pixel, expand_aggs_and_cols
        )
        extend_cpu, extend_cuda = _build_extend_area_to_line_axis1_x_constant(
            draw_trapezoid_y, expand_aggs_and_cols
        )

        x_values = self.x
        y_names = self.y
        y_stack_names = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                ys = self.to_cupy_array(df,list(y_names))
                y_stacks = self.to_cupy_array(df,list(y_stack_names))
                do_extend = extend_cuda[cuda_args(ys.shape)]
            else:
                ys = df.loc[:, list(y_names)].to_numpy()
                y_stacks = df.loc[:, list(y_stack_names)].to_numpy()
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                x_values, ys, y_stacks, *aggs_and_cols
            )

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
            np.nanmin([np.nanmin(df[c].values).item() for c in self.x]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.x])]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])

        return (self.maybe_expand_bounds(x_extents),
                self.compute_y_bounds())

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(
            append, map_onto_pixel, expand_aggs_and_cols
        )
        extend_cpu, extend_cuda = _build_extend_area_to_zero_axis1_y_constant(
            draw_trapezoid_y, expand_aggs_and_cols
        )

        x_names = self.x
        y_values = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df,list(x_names))
                ys = cp.asarray(y_values)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys = y_values
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                xs, ys, *aggs_and_cols
            )

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
            np.nanmin([np.nanmin(df[c].values).item() for c in self.x]),
            np.nanmax([np.nanmax(df[c].values).item() for c in self.x])]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])

        return (self.maybe_expand_bounds(x_extents),
                self.compute_y_bounds())

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(
            append, map_onto_pixel, expand_aggs_and_cols
        )
        extend_cpu, extend_cuda = _build_extend_area_to_line_axis1_y_constant(
            draw_trapezoid_y, expand_aggs_and_cols
        )
        x_names = self.x
        y_values = self.y
        y_stack_values = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])

            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df,list(x_names))
                ys = cp.asarray(y_values)
                ysv = cp.asarray(y_stack_values)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys = y_values
                ysv = y_stack_values
                do_extend = extend_cpu

            do_extend(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                xs, ys, ysv, *aggs_and_cols
            )

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
        bounds = self._compute_bounds(df[self.x].array.flat_array)
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds = self._compute_bounds(df[self.y].array.flat_array)

        # Make sure bounds include zero
        bounds = min(bounds[0], 0), max(bounds[1], 0)

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

        # Make sure y_extents include 0
        y_extents = min(y_extents[0], 0), max(y_extents[1], 0)

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(
            append, map_onto_pixel, expand_aggs_and_cols
        )
        perform_extend_cpu = _build_extend_area_to_zero_axis1_ragged(
            draw_trapezoid_y, expand_aggs_and_cols
        )
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds

            xs = df[x_name].values
            ys = df[y_name].values
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            perform_extend_cpu(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                xs, ys, *aggs_and_cols
            )

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
        bounds = self._compute_bounds(df[self.x].array.flat_array)
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds_y = self._compute_bounds(df[self.y].array.flat_array)
        bounds_y_stack = self._compute_bounds(
            df[self.y_stack].array.flat_array)

        # Make sure bounds include zero
        bounds = (min(bounds_y[0], bounds_y_stack[0], 0),
                  max(bounds_y[1], bounds_y_stack[1], 0))

        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin(df[self.x].array.flat_array).item(),
            np.nanmax(df[self.x].array.flat_array).item(),
            np.nanmin(df[self.y].array.flat_array).item(),
            np.nanmax(df[self.y].array.flat_array).item(),
            np.nanmin(df[self.y_stack].array.flat_array).item(),
            np.nanmax(df[self.y_stack].array.flat_array).item(),
        ]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, [2, 4]]), np.nanmax(r[:, [3, 5]])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(
            append, map_onto_pixel, expand_aggs_and_cols
        )
        extend_cpu = _build_extend_area_to_line_axis1_ragged(
            draw_trapezoid_y, expand_aggs_and_cols
        )
        x_name = self.x
        y_name = self.y
        y_stack_name = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds

            xs = df[x_name].values
            ys = df[y_name].values
            y_stacks = df[y_stack_name].values

            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            extend_cpu(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                xs, ys, y_stacks, *aggs_and_cols
            )

        return extend


def _build_draw_trapezoid_y(append, map_onto_pixel, expand_aggs_and_cols):
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
    @expand_aggs_and_cols
    def draw_trapezoid_y(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x0, x1, y0, y1, y2, y3, trapezoid_start, stacked, *aggs_and_cols
    ):
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
        x0, x1: float
            x-coordinate indices of the start and stop edge of the trapezoid
        y0, y1, y2, y3: float
            y-coordinate indices of the four corners of the trapezoid
        xmin, xmax, ymin, ymax: float
            The minimum and maximum allowable x and y value respectively.
            The trapezoid will be clipped to these values.
        i: int
            Group index
        trapezoid_start: bool
            If True, the filled trapezoid will include the (x0, y0) to (x0, y1)
            edge. Otherwise this edge will not be included.
        stacked: bool
            If False, the filled trapezoid will include the
            (x0, y1) to (x1, y2) edge. Otherwise this edge will not
            be included.
        """
        x0, x1, y0, y1, y2, y3, skip, clipped_start, clipped_end = \
            _skip_or_clip_trapezoid_y(
                x0, x1, y0, y1, y2, y3, xmin, xmax, ymin, ymax)

        if skip:
            return

        x0i, y0i = map_onto_pixel(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0, y0
        )
        _, y1i = map_onto_pixel(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0, y1
        )
        x1i, y2i = map_onto_pixel(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, x1, y2
        )
        _, y3i = map_onto_pixel(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, x1, y3
        )

        xmaxi, ymaxi = map_onto_pixel(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, xmax, ymax
        )

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
        trapezoid_start = trapezoid_start or clipped_start

        if trapezoid_start:
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
        clipped = clipped_start or clipped_end
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
def _skip_or_clip_trapezoid_y(
        x0, x1, y0, y1, y2, y3, xmin, xmax, ymin, ymax
):
    skip = False

    # If any of the coordinates are NaN, there's a discontinuity.
    # Skip the entire trapezoid.
    if (isnull(x0) or
            isnull(x1) or
            isnull(y0) or
            isnull(y1) or
            isnull(y2) or
            isnull(y3)):
        skip = True

    # Check if trapezoid is out of bounds vertically
    if ((y0 > ymax and y1 > ymax and y2 > ymax and y3 > ymax) or
            (y0 < ymin and y1 < ymin and y2 < ymin and y3 < ymin)):
        # No need to perform clipping, we will skip the whole thing
        skip = True
        clipped_start = clipped_end = False
        return x0, x1, y0, y1, y2, y3, skip, clipped_start, clipped_end

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
        clipped_end = True
        x1 = x0 + t1 * dx
        y2 = y1 + t1 * dy1
        y3 = y0 + t1 * dy0
    else:
        clipped_end = False

    if t0 > 0:
        # If x0 is clipped, we need to plot the new start
        clipped_start = True
        x0 = x0 + t0 * dx
        y0 = y0 + t0 * dy0
        y1 = y1 + t0 * dy1
    else:
        clipped_start = False

    return x0, x1, y0, y1, y2, y3, skip, clipped_start, clipped_end


def _build_extend_area_to_zero_axis0(
        draw_trapezoid_y, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                       plot_start, xs, ys, *aggs_and_cols):
        stacked = False

        x0 = xs[i]
        x1 = xs[i + 1]
        y0 = ys[i]
        y1 = 0.0
        y2 = 0.0
        y3 = ys[i + 1]
        trapezoid_start = (plot_start if i == 0 else
                           (isnull(xs[i - 1]) or isnull(ys[i - 1])))

        draw_trapezoid_y(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x0, x1, y0, y1, y2, y3, trapezoid_start, stacked, *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            plot_start, xs, ys, *aggs_and_cols
    ):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""
        nrows = xs.shape[0]
        for i in range(nrows - 1):
            perform_extend(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                           plot_start, xs, ys, *aggs_and_cols)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            plot_start, xs, ys, *aggs_and_cols
    ):
        i = cuda.grid(1)
        if i < xs.shape[0] - 1:
            perform_extend(
                i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                plot_start, xs, ys, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_area_to_line_axis0(
        draw_trapezoid_y, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, plot_start,
                       xs, ys0, ys1, *aggs_and_cols):
        x0 = xs[i]
        x1 = xs[i + 1]
        y0 = ys0[i]
        y1 = ys1[i]
        y2 = ys1[i + 1]
        y3 = ys0[i + 1]
        trapezoid_start = (plot_start if i == 0 else
                           (isnull(xs[i - 1]) or
                            isnull(ys0[i - 1]) or
                            isnull(ys1[i - 1])))
        stacked = True
        draw_trapezoid_y(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x0, x1, y0, y1, y2, y3, trapezoid_start, stacked,
            *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            plot_start, xs, ys0, ys1, *aggs_and_cols
    ):
        """Aggregate filled area between the line formed by
        ``xs`` and ``ys0`` and the line formed by ``xs`` and ``ys1``"""
        nrows = xs.shape[0]
        for i in range(nrows - 1):
            perform_extend(
                i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                plot_start, xs, ys0, ys1, *aggs_and_cols
            )

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            plot_start, xs, ys0, ys1, *aggs_and_cols
    ):
        i = cuda.grid(1)
        if i < xs.shape[0] - 1:
            perform_extend(
                i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                plot_start, xs, ys0, ys1, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_area_to_zero_axis0_multi(
        draw_trapezoid_y, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                       plot_start, xs, ys, *aggs_and_cols):
        x0 = xs[i, j]
        x1 = xs[i + 1, j]
        y0 = ys[i, j]
        y1 = 0.0
        y2 = 0.0
        y3 = ys[i + 1, j]
        trapezoid_start = (plot_start if i == 0 else
                           (isnull(xs[i - 1, j]) or
                            isnull(ys[i - 1, j])))
        stacked = False
        draw_trapezoid_y(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x0, x1, y0, y1, y2, y3, trapezoid_start, stacked,
            *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            plot_start, xs, ys, *aggs_and_cols
    ):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""

        nrows, ncols = xs.shape
        for j in range(ncols):
            for i in range(nrows - 1):
                perform_extend(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                               plot_start, xs, ys, *aggs_and_cols)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            plot_start, xs, ys, *aggs_and_cols
    ):
        i, j = cuda.grid(2)
        if i < xs.shape[0] - 1 and j < xs.shape[1]:
            perform_extend(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                plot_start, xs, ys, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_area_to_line_axis0_multi(
        draw_trapezoid_y, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                       plot_start, xs, ys0, ys1, *aggs_and_cols):
        x0 = xs[i, j]
        x1 = xs[i + 1, j]
        y0 = ys0[i, j]
        y1 = ys1[i, j]
        y2 = ys1[i + 1, j]
        y3 = ys0[i + 1, j]
        trapezoid_start = (plot_start if i == 0 else
                           (isnull(xs[i - 1, j]) or
                            isnull(ys0[i - 1, j]) or
                            isnull(ys1[i - 1, j])))
        stacked = True
        draw_trapezoid_y(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x0, x1, y0, y1, y2, y3, trapezoid_start, stacked,
            *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            plot_start, xs, ys0, ys1, *aggs_and_cols
    ):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""

        nrows, ncols = xs.shape

        for j in range(ncols):
            for i in range(nrows - 1):
                perform_extend(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                               plot_start, xs, ys0, ys1, *aggs_and_cols)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            plot_start, xs, ys0, ys1, *aggs_and_cols
    ):
        i, j = cuda.grid(2)
        if i < xs.shape[0] - 1 and j < xs.shape[1]:
            perform_extend(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                plot_start, xs, ys0, ys1, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_area_to_zero_axis1_none_constant(
        draw_trapezoid_y, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend(
            i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys, *aggs_and_cols
    ):
        x0 = xs[i, j]
        x1 = xs[i, j + 1]

        y0 = ys[i, j]
        y1 = 0.0
        y2 = 0.0
        y3 = ys[i, j + 1]

        trapezoid_start = (j == 0 or
                           isnull(xs[i, j - 1]) or
                           isnull(ys[i, j - 1]))
        stacked = False
        draw_trapezoid_y(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x0, x1, y0, y1, y2, y3, trapezoid_start, stacked, *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
    ):
        nrows, ncols = xs.shape
        for i in range(nrows):
            for j in range(ncols - 1):
                perform_extend(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys, *aggs_and_cols
                )

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
    ):
        i, j = cuda.grid(2)
        if i < xs.shape[0] and j < xs.shape[1] - 1:
            perform_extend(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                xs, ys, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_area_to_line_axis1_none_constant(
        draw_trapezoid_y, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend(
            i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys0, ys1, *aggs_and_cols
    ):
        x0 = xs[i, j]
        x1 = xs[i, j + 1]

        y0 = ys0[i, j]
        y1 = ys1[i, j]
        y2 = ys1[i, j + 1]
        y3 = ys0[i, j + 1]

        trapezoid_start = (j == 0 or
                           isnull(xs[i, j - 1]) or
                           isnull(ys0[i, j - 1]) or
                           isnull(ys1[i, j - 1]))
        stacked = True
        draw_trapezoid_y(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x0, x1, y0, y1, y2, y3, trapezoid_start, stacked,
            *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys0, ys1, *aggs_and_cols
    ):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""
        nrows, ncols = xs.shape
        for i in range(nrows):
            for j in range(ncols - 1):
                perform_extend(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys0, ys1, *aggs_and_cols
                )

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys0, ys1, *aggs_and_cols
    ):
        i, j = cuda.grid(2)
        if i < xs.shape[0] and j < xs.shape[1] - 1:
            perform_extend(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                xs, ys0, ys1, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_area_to_zero_axis1_x_constant(
        draw_trapezoid_y, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                       xs, ys, *aggs_and_cols):

        x0 = xs[j]
        x1 = xs[j + 1]

        y0 = ys[i, j]
        y1 = 0.0
        y2 = 0.0
        y3 = ys[i, j + 1]

        trapezoid_start = (j == 0 or
                           isnull(xs[j - 1]) or
                           isnull(ys[i, j - 1]))
        stacked = False
        draw_trapezoid_y(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x0, x1, y0, y1, y2, y3, trapezoid_start, stacked,
            *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
    ):
        nrows, ncols = ys.shape
        for i in range(nrows):
            for j in range(ncols - 1):
                perform_extend(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys, *aggs_and_cols
                )

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
    ):
        i, j = cuda.grid(2)
        if i < ys.shape[0] and j < ys.shape[1] - 1:
            perform_extend(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                xs, ys, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_area_to_line_axis1_x_constant(
        draw_trapezoid_y, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend(
            i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys0, ys1, *aggs_and_cols
    ):
        x0 = xs[j]
        x1 = xs[j + 1]

        y0 = ys0[i, j]
        y1 = ys1[i, j]
        y2 = ys1[i, j + 1]
        y3 = ys0[i, j + 1]

        trapezoid_start = (j == 0 or
                           isnull(xs[j - 1]) or
                           isnull(ys0[i, j - 1]) or
                           isnull(ys1[i, j - 1]))
        stacked = True
        draw_trapezoid_y(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x0, x1, y0, y1, y2, y3, trapezoid_start, stacked, *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys0, ys1, *aggs_and_cols
    ):
        nrows, ncols = ys0.shape
        for i in range(nrows):
            for j in range(ncols - 1):
                perform_extend(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys0, ys1, *aggs_and_cols
                )

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys0, ys1, *aggs_and_cols
    ):
        i, j = cuda.grid(2)
        if i < ys0.shape[0] and j < ys0.shape[1] - 1:
            perform_extend(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                xs, ys0, ys1, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_area_to_zero_axis1_y_constant(
        draw_trapezoid_y, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend(
            i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys, *aggs_and_cols
    ):
        x0 = xs[i, j]
        x1 = xs[i, j + 1]

        y0 = ys[j]
        y1 = 0.0
        y2 = 0.0
        y3 = ys[j + 1]

        trapezoid_start = (j == 0 or
                           isnull(xs[i, j - 1]) or
                           isnull(ys[j - 1]))
        stacked = False
        draw_trapezoid_y(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x0, x1, y0, y1, y2, y3, trapezoid_start, stacked,
            *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
    ):
        nrows, ncols = xs.shape
        for i in range(nrows):
            for j in range(ncols - 1):
                perform_extend(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys, *aggs_and_cols
                )

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
    ):
        i, j = cuda.grid(2)
        if i < xs.shape[0] and j < xs.shape[1] - 1:
            perform_extend(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                xs, ys, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_area_to_line_axis1_y_constant(
        draw_trapezoid_y, expand_aggs_and_cols
):
    @ngjit
    @expand_aggs_and_cols
    def perform_extend(
            i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys0, ys1, *aggs_and_cols
    ):
        x0 = xs[i, j]
        x1 = xs[i, j + 1]

        y0 = ys0[j]
        y1 = ys1[j]
        y2 = ys1[j + 1]
        y3 = ys0[j + 1]

        trapezoid_start = (j == 0 or
                           isnull(xs[i, j - 1]) or
                           isnull(ys0[j - 1]) or
                           isnull(ys1[j - 1]))
        stacked = True
        draw_trapezoid_y(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x0, x1, y0, y1, y2, y3, trapezoid_start, stacked,
            *aggs_and_cols
        )


    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys0, ys1, *aggs_and_cols
    ):
        nrows, ncols = xs.shape
        for i in range(nrows):
            for j in range(ncols - 1):
                perform_extend(
                    i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    xs, ys0, ys1, *aggs_and_cols
                )

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys0, ys1, *aggs_and_cols
    ):
        i, j = cuda.grid(2)
        if i < xs.shape[0] and j < xs.shape[1] - 1:
            perform_extend(
                i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                xs, ys0, ys1, *aggs_and_cols
            )

    return extend_cpu, extend_cuda


def _build_extend_area_to_zero_axis1_ragged(
        draw_trapezoid_y, expand_aggs_and_cols
):
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols
    ):
        x_start_inds = xs.start_indices
        x_flat = xs.flat_array

        y_start_inds = ys.start_indices
        y_flat = ys.flat_array

        perform_extend_area_to_zero_axis1_ragged(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x_start_inds, x_flat, y_start_inds, y_flat,
            *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_area_to_zero_axis1_ragged(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x_start_inds, x_flat, y_start_inds, y_flat,
            *aggs_and_cols
    ):
        nrows = len(x_start_inds)
        x_flat_len = len(x_flat)
        y_flat_len = len(y_flat)

        i = 0
        while i < nrows:
            # Get x index range
            x_start_i = x_start_inds[i]
            x_stop_i = (x_start_inds[i + 1] if i < nrows - 1 else x_flat_len)

            # Get y index range
            y_start_i = y_start_inds[i]
            y_stop_i = (y_start_inds[i + 1] if i < nrows - 1 else y_flat_len)

            # Find line segment length as shorter of the two segments
            segment_len = min(x_stop_i - x_start_i,
                              y_stop_i - y_start_i)

            j = 0
            while j < segment_len - 1:

                x0 = x_flat[x_start_i + j]
                x1 = x_flat[x_start_i + j + 1]

                y0 = y_flat[y_start_i + j]
                y1 = 0.0
                y2 = 0.0
                y3 = y_flat[y_start_i + j + 1]

                trapezoid_start = (j == 0 or
                                   isnull(x_flat[x_start_i + j - 1]) or
                                   isnull(y_flat[y_start_i + j - 1]))
                stacked = False
                draw_trapezoid_y(
                    i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    x0, x1, y0, y1, y2, y3, trapezoid_start, stacked,
                    *aggs_and_cols
                )

                j += 1
            i += 1

    return extend_cpu


def _build_extend_area_to_line_axis1_ragged(
        draw_trapezoid_y, expand_aggs_and_cols
):
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            xs, ys0, ys1, *aggs_and_cols
    ):
        x_start_inds = xs.start_indices
        x_flat = xs.flat_array

        y0_start_inds = ys0.start_indices
        y0_flat = ys0.flat_array

        y1_start_inds = ys1.start_indices
        y1_flat = ys1.flat_array

        perform_extend_area_to_line_axis1_ragged(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, x_start_inds, x_flat,
            y0_start_inds, y0_flat, y1_start_inds, y1_flat,
            *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_area_to_line_axis1_ragged(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            x_start_inds, x_flat, y0_start_inds, y0_flat, y1_start_inds,
            y1_flat, *aggs_and_cols):

        nrows = len(x_start_inds)
        x_flat_len = len(x_flat)
        y0_flat_len = len(y0_flat)
        y1_flat_len = len(y1_flat)

        i = 0
        while i < nrows:
            # Get x index range
            x_start_i = x_start_inds[i]
            x_stop_i = (x_start_inds[i + 1] if i < nrows - 1 else x_flat_len)

            # Get y index range
            y0_start_i = y0_start_inds[i]
            y0_stop_i = (y0_start_inds[i + 1] if i < nrows - 1 else y0_flat_len)

            y1_start_i = y1_start_inds[i]
            y1_stop_i = (y1_start_inds[i + 1] if i < nrows - 1 else y1_flat_len)

            # Find line segment length as shorter of the two segments
            segment_len = min(x_stop_i - x_start_i,
                              y0_stop_i - y0_start_i,
                              y1_stop_i - y1_start_i)

            j = 0
            while j < segment_len - 1:
                x0 = x_flat[x_start_i + j]
                x1 = x_flat[x_start_i + j + 1]

                y0 = y0_flat[y0_start_i + j]
                y1 = y1_flat[y1_start_i + j]
                y2 = y1_flat[y1_start_i + j + 1]
                y3 = y0_flat[y0_start_i + j + 1]

                trapezoid_start = (j == 0 or
                                   isnull(x_flat[x_start_i + j - 1]) or
                                   isnull(y0_flat[y0_start_i + j - 1]) or
                                   isnull(y1_flat[y1_start_i + j] - 1))
                stacked = True
                draw_trapezoid_y(
                    i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                    x0, x1, y0, y1, y2, y3, trapezoid_start, stacked,
                    *aggs_and_cols
                )
                j += 1
            i += 1

    return extend_cpu
