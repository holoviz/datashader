from __future__ import absolute_import, division

from distutils.version import LooseVersion

import dask
import pandas as pd
import dask.dataframe as dd
from dask.base import tokenize, compute
from dask.context import _globals

from .core import bypixel
from .compatibility import apply
from .compiler import compile_components
from .glyphs import Glyph, Line
from .utils import Dispatcher

__all__ = ()


@bypixel.pipeline.register(dd.DataFrame)
def dask_pipeline(df, schema, canvas, glyph, summary):
    dsk, name = glyph_dispatch(glyph, df, schema, canvas, summary)

    if LooseVersion(dask.__version__) >= '0.18.0':
        get = dask.base.get_scheduler() or df.__dask_scheduler__
    else:
        get = _globals.get('get') or getattr(df, '__dask_scheduler__', None) or df._default_get
    keys = getattr(df, '__dask_keys__', None) or df._keys
    optimize = getattr(df, '__dask_optimize__', None) or df._optimize

    dsk.update(optimize(df.dask, keys()))

    return get(dsk, name)


def shape_bounds_st_and_axis(df, canvas, glyph):
    x_range = canvas.x_range or glyph._compute_x_bounds_dask(df)
    y_range = canvas.y_range or glyph._compute_y_bounds_dask(df)
    x_min, x_max, y_min, y_max = bounds = compute(*(x_range + y_range))
    x_range, y_range = (x_min, x_max), (y_min, y_max)

    width = canvas.plot_width
    height = canvas.plot_height

    x_st = canvas.x_axis.compute_scale_and_translate(x_range, width)
    y_st = canvas.y_axis.compute_scale_and_translate(y_range, height)
    st = x_st + y_st
    shape = (height, width)

    x_axis = canvas.x_axis.compute_index(x_st, width)
    y_axis = canvas.y_axis.compute_index(y_st, height)
    axis = [y_axis, x_axis]

    return shape, bounds, st, axis


glyph_dispatch = Dispatcher()


@glyph_dispatch.register(Glyph)
def default(glyph, df, schema, canvas, summary):
    shape, bounds, st, axis = shape_bounds_st_and_axis(df, canvas, glyph)

    # Compile functions
    create, info, append, combine, finalize = \
        compile_components(summary, schema, glyph)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(x_mapper, y_mapper, info, append)

    def chunk(df):
        aggs = create(shape)
        extend(aggs, df, st, bounds)
        return aggs

    name = tokenize(df._name, canvas, glyph, summary)
    dask_dataframe_keys = getattr(df, '__dask_keys__', None) or df._keys
    keys = dask_dataframe_keys()
    keys2 = [(name, i) for i in range(len(keys))]
    dsk = dict((k2, (chunk, k)) for (k2, k) in zip(keys2, keys))
    dsk[name] = (apply, finalize, [(combine, keys2)],
                 dict(coords=axis, dims=[glyph.y, glyph.x]))
    return dsk, name


@glyph_dispatch.register(Line)
def line(glyph, df, schema, canvas, summary):
    shape, bounds, st, axis = shape_bounds_st_and_axis(df, canvas, glyph)

    # Compile functions
    create, info, append, combine, finalize = \
        compile_components(summary, schema, glyph)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(x_mapper, y_mapper, info, append)

    def chunk(df, df2=None):
        plot_start = True
        if df2 is not None:
            df = pd.concat([df.iloc[-1:], df2])
            plot_start = False
        aggs = create(shape)
        extend(aggs, df, st, bounds, plot_start=plot_start)
        return aggs

    name = tokenize(df._name, canvas, glyph, summary)
    old_name = df._name
    dsk = {(name, 0): (chunk, (old_name, 0))}
    for i in range(1, df.npartitions):
        dsk[(name, i)] = (chunk, (old_name, i - 1), (old_name, i))
    keys2 = [(name, i) for i in range(df.npartitions)]
    dsk[name] = (apply, finalize, [(combine, keys2)],
                 dict(coords=axis, dims=[glyph.y, glyph.x]))
    return dsk, name
