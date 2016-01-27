from __future__ import absolute_import, division

import pandas as pd

from .core import pipeline
from .compiler import compile_components
from .glyphs import compute_x_bounds, compute_y_bounds

__all__ = ()


@pipeline.register(pd.DataFrame)
def pandas_pipeline(df, schema, canvas, glyph, summary):
    create, info, append, _, finalize = compile_components(summary, schema)
    extend = glyph._build_extend(info, append)

    bases = create((canvas.plot_height, canvas.plot_width))
    x_range = canvas.x_range or compute_x_bounds(glyph, df)
    y_range = canvas.y_range or compute_y_bounds(glyph, df)
    vt = canvas.view_transform(x_range, y_range)
    extend(bases, df, vt, x_range + y_range)
    return finalize(bases,
                    x_axis=canvas.x_axis_type(x_range),
                    y_axis=canvas.y_axis_type(y_range))
