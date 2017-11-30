from __future__ import absolute_import, division

import numpy as np
import pandas as pd

from .core import bypixel
from .compiler import compile_components
from .glyphs import _PointLike, _PolygonLike
from .utils import Dispatcher

__all__ = ()


@bypixel.pipeline.register(pd.DataFrame)
def pandas_pipeline(df, schema, canvas, glyph, summary):
    return glyph_dispatch(glyph, df, schema, canvas, summary)


glyph_dispatch = Dispatcher()


@glyph_dispatch.register(_PointLike)
def pointlike(glyph, df, schema, canvas, summary):
    create, info, append, _, finalize = compile_components(summary, schema, glyph)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(x_mapper, y_mapper, info, append)

    x_range = canvas.x_range or glyph._compute_x_bounds(df[glyph.x].values)
    y_range = canvas.y_range or glyph._compute_y_bounds(df[glyph.y].values)

    width = canvas.plot_width
    height = canvas.plot_height

    x_st = canvas.x_axis.compute_scale_and_translate(x_range, width)
    y_st = canvas.y_axis.compute_scale_and_translate(y_range, height)

    x_axis = canvas.x_axis.compute_index(x_st, width)
    y_axis = canvas.y_axis.compute_index(y_st, height)

    bases = create((height, width))
    extend(bases, df, x_st + y_st, x_range + y_range)

    return finalize(bases, coords=[y_axis, x_axis], dims=[glyph.y, glyph.x])



@glyph_dispatch.register(_PolygonLike)
def polygonlike(glyph, df, schema, canvas, summary):
    create, info, append, _, finalize = compile_components(summary, schema, glyph)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(x_mapper, y_mapper, info, append)

    xs = df[glyph.x].values
    x_range = canvas.x_range or glyph._compute_x_bounds(xs.reshape(np.prod(xs.shape)))
    ys = df[glyph.y].values
    y_range = canvas.y_range or glyph._compute_y_bounds(ys.reshape(np.prod(ys.shape)))

    width = canvas.plot_width
    height = canvas.plot_height

    x_st = canvas.x_axis.compute_scale_and_translate(x_range, width)
    y_st = canvas.y_axis.compute_scale_and_translate(y_range, height)

    x_axis = canvas.x_axis.compute_index(x_st, width)
    y_axis = canvas.y_axis.compute_index(y_st, height)

    bases = create((height, width))
    extend(bases, df, x_st + y_st, x_range + y_range, weight_type=glyph.weight_type, interpolate=glyph.interpolate)

    return finalize(bases, coords=[y_axis, x_axis], dims=[glyph.y, glyph.x])
