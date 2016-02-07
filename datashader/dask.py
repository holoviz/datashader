from __future__ import absolute_import, division

import dask.dataframe as dd
from dask.base import tokenize, compute
from dask.context import _globals

from .core import pipeline
from .compatibility import apply
from .compiler import compile_components
from .glyphs import compute_x_bounds, compute_y_bounds

__all__ = ()


@pipeline.register(dd.DataFrame)
def dask_pipeline(df, schema, canvas, glyph, summary):
    create, info, append, combine, finalize = compile_components(summary,
                                                                 schema)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(x_mapper, y_mapper, info, append)

    x_range = canvas.x_range or compute_x_bounds(glyph, df)
    y_range = canvas.y_range or compute_y_bounds(glyph, df)
    x_min, x_max, y_min, y_max = bounds = compute(*(x_range + y_range))
    x_range, y_range = (x_min, x_max), (y_min, y_max)
    width = canvas.plot_width
    height = canvas.plot_height

    x_st = canvas.x_axis.scale_and_translation(x_range, width)
    y_st = canvas.y_axis.scale_and_translation(y_range, height)
    st = x_st + y_st
    shape = (height, width)

    x_axis = canvas.x_axis.compute_index(width, x_st)
    y_axis = canvas.y_axis.compute_index(height, y_st)

    def chunk(df):
        aggs = create(shape)
        extend(aggs, df, st, bounds)
        return aggs

    name = tokenize(df._name, canvas, glyph, summary)
    keys = df._keys()
    keys2 = [(name, i) for i in range(len(keys))]
    dsk = dict((k2, (chunk, k)) for (k2, k) in zip(keys2, keys))
    dsk[name] = (apply, finalize, [(combine, keys2)],
                 dict(coords=[y_axis, x_axis], dims=['y_axis', 'x_axis']))
    dsk.update(df.dask)
    dsk = df._optimize(dsk, name)

    get = _globals['get'] or df._default_get

    return get(dsk, name)
