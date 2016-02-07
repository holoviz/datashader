from __future__ import absolute_import, division

import pandas as pd

from .core import pipeline
from .compiler import compile_components
from .glyphs import compute_x_bounds, compute_y_bounds

__all__ = ()


@pipeline.register(pd.DataFrame)
def pandas_pipeline(df, schema, canvas, glyph, summary):
    create, info, append, _, finalize = compile_components(summary, schema)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(x_mapper, y_mapper, info, append)

    x_range = canvas.x_range or compute_x_bounds(glyph, df)
    y_range = canvas.y_range or compute_y_bounds(glyph, df)
    width = canvas.plot_width
    height = canvas.plot_height

    x_st = canvas.x_axis.scale_and_translation(x_range, width)
    y_st = canvas.y_axis.scale_and_translation(y_range, height)

    x_axis = canvas.x_axis.compute_index(width, x_st)
    y_axis = canvas.y_axis.compute_index(height, y_st)

    bases = create((height, width))
    extend(bases, df, x_st + y_st, x_range + y_range)

    return finalize(bases, coords=[y_axis, x_axis], dims=['y_axis', 'x_axis'])
