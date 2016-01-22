from __future__ import absolute_import, division

import dask.dataframe as dd
from dask.base import tokenize, compute
from dask.context import _globals

from .core import pipeline
from .compiler import compile_components
from .glyphs import compute_x_bounds, compute_y_bounds

__all__ = ()


@pipeline.register(dd.DataFrame)
def dask_pipeline(df, schema, canvas, glyph, summary):
    create, info, append, combine, finalize = compile_components(summary,
                                                                 schema)
    extend = glyph._build_extend(info, append)

    x_range = canvas.x_range or compute_x_bounds(glyph, df)
    y_range = canvas.y_range or compute_y_bounds(glyph, df)
    x_min, x_max, y_min, y_max = bounds = compute(*(x_range + y_range))
    x_range, y_range = (x_min, x_max), (y_min, y_max)

    vt = canvas.view_transform(x_range, y_range)
    shape = (canvas.plot_height, canvas.plot_width)

    def chunk(df):
        aggs = create(shape)
        extend(aggs, df, vt, bounds)
        return aggs

    name = tokenize(df._name, canvas, glyph, summary)
    keys = df._keys()
    keys2 = [(name, i) for i in range(len(keys))]
    dsk = dict((k2, (chunk, k)) for (k2, k) in zip(keys2, keys))
    dsk[name] = (finalize, (combine, keys2))
    dsk.update(df.dask)
    dsk = df._optimize(dsk, name)

    get = _globals['get'] or df._default_get

    return get(dsk, name)
