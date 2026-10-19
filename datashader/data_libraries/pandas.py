from __future__ import annotations

import pandas as pd

from datashader.core import bypixel
from datashader.compiler import compile_components
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.glyphs.area import _AreaToLineLike
from datashader.glyphs.line import LinesXarrayCommonX
from datashader.utils import Dispatcher

__all__ = ()


@bypixel.pipeline.register(pd.DataFrame)
def pandas_pipeline(df, schema, canvas, glyph, summary, *, antialias=False):
    return glyph_dispatch(glyph, df, schema, canvas, summary, antialias=antialias)


glyph_dispatch = Dispatcher()


@glyph_dispatch.register(_PointLike)
@glyph_dispatch.register(_GeometryLike)
@glyph_dispatch.register(_AreaToLineLike)
def default(glyph, source, schema, canvas, summary, *, antialias=False, cuda=False):
    create, info, append, _, finalize, antialias_stage_2, antialias_stage_2_funcs, _ = \
        compile_components(summary, schema, glyph, antialias=antialias, cuda=cuda,
                           partitioned=False)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(
        x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs)

    x_range = canvas.x_range or glyph.compute_x_bounds(source)
    y_range = canvas.y_range or glyph.compute_y_bounds(source)
    canvas.validate_ranges(x_range, y_range)

    width = canvas.plot_width
    height = canvas.plot_height

    x_st = canvas.x_axis.compute_scale_and_translate(x_range, width)
    y_st = canvas.y_axis.compute_scale_and_translate(y_range, height)

    x_axis = canvas.x_axis.compute_index(x_st, width)
    y_axis = canvas.y_axis.compute_index(y_st, height)

    bases = create((height, width))

    if isinstance(glyph, LinesXarrayCommonX) and summary.uses_row_index(cuda, partitioned=False):
        # Need to use a row index and extract.apply() doesn't have enough
        # information to determine the coordinate length itself so do so here
        # and pass it along as an xarray attribute in the usual manner.
        other_dim_index = 1 - glyph.x_dim_index
        other_dim_name = source[glyph.y].coords.dims[other_dim_index]
        length = len(source[other_dim_name])
        source = source.assign_attrs(_datashader_row_offset=0, _datashader_row_length=length)

    extend(bases, source, x_st + y_st, x_range + y_range)

    return finalize(bases,
                    cuda=cuda,
                    coords=dict([(glyph.x_label, x_axis),
                                 (glyph.y_label, y_axis)]),
                    dims=[glyph.y_label, glyph.x_label],
                    attrs=dict(x_range=x_range, y_range=y_range))
