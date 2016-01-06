from __future__ import absolute_import, division

from blaze import compute, dispatch
import pandas as pd

from ..expr import ByPixel, Canvas
from .glyph import (subselect, make_extend, view_transform, get_x_range,
                    get_y_range)
from .compiler import compile_components

__all__ = ['compute_down', 'optimize']


@dispatch(ByPixel, pd.DataFrame)
def compute_down(expr, data):
    df = compute(expr._child, data)

    create, info, append, combine, finalize = compile_components(expr)
    extend = make_extend(expr.glyph, info, append)

    aggs = create((expr.canvas.plot_height, expr.canvas.plot_width))
    vt = view_transform(expr.canvas)
    extend(aggs, df, vt)
    return finalize(aggs)


@dispatch(ByPixel, pd.DataFrame)
def optimize(expr, data):
    x_range = y_range = None
    if not expr.canvas.x_range:
        xmin, xmax = get_x_range(expr.glyph)
        x_range = compute(xmin, data), compute(xmax, data)
    if not expr.canvas.y_range:
        ymin, ymax = get_y_range(expr.glyph)
        y_range = compute(ymin, data), compute(ymax, data)
    lhs = expr._child
    rhs = subselect(expr.glyph, df=lhs,
                    x_range=expr.canvas.x_range,
                    y_range=expr.canvas.y_range)
    expr = expr._subs({lhs: rhs})
    if x_range is not None or y_range is not None:
        c = expr.canvas
        canvas = Canvas(plot_width=c.plot_width,
                        plot_height=c.plot_height,
                        x_range=x_range,
                        y_range=y_range,
                        stretch=c.stretch)
        expr.canvas = canvas
    return expr
