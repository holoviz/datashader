from __future__ import absolute_import, division

import pandas as pd

from .core import bypixel
from .compiler import compile_components

__all__ = ()


@bypixel.pipeline.register(pd.DataFrame)
def pandas_pipeline(df, schema, canvas, glyph, summary):
    create, info, append, _, finalize = compile_components(summary, schema)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(x_mapper, y_mapper, info, append)

    x_range = canvas.x_range or glyph._compute_x_bounds(df[glyph.x].values.ravel())
    y_range = canvas.y_range or glyph._compute_y_bounds(df[glyph.y].values.ravel())

    width = canvas.plot_width
    height = canvas.plot_height

    x_st = canvas.x_axis.compute_scale_and_translate(x_range, width)
    y_st = canvas.y_axis.compute_scale_and_translate(y_range, height)

    x_axis = canvas.x_axis.compute_index(x_st, width)
    y_axis = canvas.y_axis.compute_index(y_st, height)

    bases = create((height, width))
    extend(bases, df, x_st + y_st, x_range + y_range)

    if hasattr(glyph, 'xs') and hasattr(glyph, 'ys'):
        return finalize(bases, coords=[y_axis, x_axis], dims=['y', 'x'])
    return finalize(bases, coords=[y_axis, x_axis], dims=[glyph.y, glyph.x])
