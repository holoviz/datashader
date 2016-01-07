from __future__ import absolute_import, division

import pandas as pd

from .core import pipeline
from .compiler import compile_components
from .dispatch import dispatch
from .glyphs import Point

__all__ = ()


@pipeline.register(pd.DataFrame)
def pandas_pipeline(df, schema, canvas, glyph, summary):
    create, info, append, _, finalize = compile_components(summary, schema)
    extend = glyph._build_extend(info, append)

    aggs = create((canvas.plot_height, canvas.plot_width))
    x_range = canvas.x_range or compute_x_bounds(glyph, df)
    y_range = canvas.y_range or compute_y_bounds(glyph, df)
    vt = canvas.view_transform(x_range, y_range)
    df = subselect(glyph, df, canvas)
    extend(aggs, df, vt)
    return finalize(aggs)


@dispatch(Point, pd.DataFrame, object)
def subselect(glyph, df, canvas):
    select = None
    if canvas.x_range:
        xmin, xmax = canvas.x_range
        x = df[glyph.x]
        select = (x >= xmin) & (x <= xmax)
    if canvas.y_range:
        ymin, ymax = canvas.y_range
        y = df[glyph.y]
        temp = (y >= ymin) & (y <= ymax)
        select = temp if select is None else temp & select
    if select is None:
        return df
    return df[select]


@dispatch(Point, pd.DataFrame)
def compute_x_bounds(glyph, df):
    return df[glyph.x].min(), df[glyph.x].max()


@dispatch(Point, pd.DataFrame)
def compute_y_bounds(glyph, df):
    return df[glyph.y].min(), df[glyph.y].max()
