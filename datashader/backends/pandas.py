from __future__ import absolute_import, division

from blaze import compute, dispatch
import pandas as pd

from ..expr import ByPixel
from .glyph import subselect, make_extend, view_transform
from .compiler import compile_components

__all__ = ['compute_down', 'optimize']


@dispatch(ByPixel, pd.DataFrame)
def compute_down(expr, data):
    df = compute(expr._child, data)

    create, info, append, finalize = compile_components(expr)
    extend = make_extend(expr.glyph, info, append)

    aggs = create((expr.canvas.plot_height, expr.canvas.plot_width))
    vt = view_transform(expr.canvas)
    extend(aggs, df, vt)
    return finalize(aggs)


@dispatch(ByPixel, pd.DataFrame)
def optimize(expr, _):
    lhs = expr._child
    rhs = subselect(expr.glyph, df=lhs,
                    x_range=expr.canvas.x_range,
                    y_range=expr.canvas.y_range)
    return expr._subs({lhs: rhs})
