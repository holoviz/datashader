from __future__ import absolute_import, division

import pandas as pd

from .core import pipeline
from .compiler import compile_components
from .glyphs import compute_x_bounds, compute_y_bounds

__all__ = ()


@pipeline.register(pd.DataFrame)
def pandas_pipeline(df, schema, canvas, glyph, summary):
    create, info, append, _, finalize = compile_components(summary, schema)
    x_mapper = canvas.x_axis_type.mapper
    y_mapper = canvas.x_axis_type.mapper
    extend = glyph._build_extend(x_mapper, y_mapper, info, append)

    x_range = canvas.x_range or compute_x_bounds(glyph, df)
    y_range = canvas.y_range or compute_y_bounds(glyph, df)
    x_axis = canvas.x_axis_type(x_range)
    y_axis = canvas.y_axis_type(y_range)

    xvt = x_axis.view_transform(canvas.plot_width)
    yvt = y_axis.view_transform(canvas.plot_height)

    bases = create((canvas.plot_height, canvas.plot_width))
    extend(bases, df, xvt + yvt, x_range + y_range)

    return finalize(bases, x_axis=x_axis, y_axis=y_axis)
