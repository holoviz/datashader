from __future__ import annotations


import numpy as np
import pandas as pd
import dask
import dask.array as da
import dask.dataframe as dd
from dask.base import tokenize, compute

from datashader.core import bypixel
from datashader.utils import apply
from datashader.compiler import compile_components
from datashader.glyphs import Glyph, LineAxis0
from datashader.utils import Dispatcher

__all__ = ()


def _dask_compat(df):
    """
    Places where this is done, are to be compatible with both
    `dask-expr` and classic `dask.dataframe` (where `optimize` does not exist).
    With dask-expr calling df.__dask_graph__() or df.__dask_keys__() will
    make the graph no longer match the df._name, so we preemptively call it
    to make it match.

    For more information, see the following comment:
    https://github.com/holoviz/datashader/pull/1317#issuecomment-2039986852
    """
    return getattr(df, 'optimize', lambda: df)()


def dask_pipeline(df, schema, canvas, glyph, summary, *, antialias=False, cuda=False):
    dsk, name = glyph_dispatch(glyph, df, schema, canvas, summary, antialias=antialias, cuda=cuda)

    # Get user configured scheduler (if any), or fall back to default
    # scheduler for dask DataFrame
    scheduler = dask.base.get_scheduler() or df.__dask_scheduler__

    if isinstance(dsk, da.Array):
        return da.compute(dsk, scheduler=scheduler)[0]

    df = _dask_compat(df)
    keys = df.__dask_keys__()
    optimize = df.__dask_optimize__
    graph = df.__dask_graph__()

    dsk.update(optimize(graph, keys))
    return scheduler(dsk, name)


bypixel.pipeline.register(dd.DataFrame)(dask_pipeline)


def shape_bounds_st_and_axis(df, canvas, glyph):
    if not canvas.x_range or not canvas.y_range:
        x_extents, y_extents = glyph.compute_bounds_dask(df)
    else:
        x_extents, y_extents = None, None

    x_range = canvas.x_range or x_extents
    y_range = canvas.y_range or y_extents
    x_min, x_max, y_min, y_max = bounds = compute(*(x_range + y_range))
    x_range, y_range = (x_min, x_max), (y_min, y_max)
    canvas.validate_ranges(x_range, y_range)

    width = canvas.plot_width
    height = canvas.plot_height

    x_st = canvas.x_axis.compute_scale_and_translate(x_range, width)
    y_st = canvas.y_axis.compute_scale_and_translate(y_range, height)
    st = x_st + y_st
    shape = (height, width)

    x_axis = canvas.x_axis.compute_index(x_st, width)
    y_axis = canvas.y_axis.compute_index(y_st, height)
    axis = dict([(glyph.x_label, x_axis), (glyph.y_label, y_axis)])

    return shape, bounds, st, axis


glyph_dispatch = Dispatcher()


@glyph_dispatch.register(Glyph)
def default(glyph, df, schema, canvas, summary, *, antialias=False, cuda=False):
    shape, bounds, st, axis = shape_bounds_st_and_axis(df, canvas, glyph)

    # Compile functions
    partitioned = isinstance(df, dd.DataFrame) and df.npartitions > 1
    create, info, append, combine, finalize, antialias_stage_2, antialias_stage_2_funcs, \
        column_names = compile_components(summary, schema, glyph, antialias=antialias, cuda=cuda,
                                          partitioned=partitioned)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(
        x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs)
    x_range = bounds[:2]
    y_range = bounds[2:]

    if summary.uses_row_index(cuda, partitioned):
        def func(partition: pd.DataFrame, cumulative_lens, partition_info=None):
            # This function is called once for each dask dataframe partition.
            # It sets the _datashader_row_offset attribute so that row indexes
            # can be calculated correctly in the reductions.extract class.
            if partition_info is not None:
                partition_index = partition_info["number"]
                row_offset = cumulative_lengths[partition_index-1] if partition_index > 0 else 0
                # Try to add new attribute to attrs if they exist, otherwise
                # just set the attribute directly.
                attrs = getattr(partition, "attrs", None)
                setattr(attrs or partition, "_datashader_row_offset", row_offset)
            return partition

        cumulative_lengths = df.map_partitions(len).compute().cumsum().to_numpy()
        df = df.map_partitions(func, cumulative_lengths)

    # Here be dragons
    # Get the dataframe graph
    df = _dask_compat(df)
    graph = df.__dask_graph__()

    # Guess a reasonable output dtype from combination of dataframe dtypes
    # Only consider columns used, not all columns in dataframe (issue #1235)
    dtypes = []

    for dt in df.dtypes[column_names]:
        if isinstance(dt, pd.CategoricalDtype):
            continue
        elif isinstance(dt, pd.api.extensions.ExtensionDtype):
            # RaggedArray implementation and
            # https://github.com/pandas-dev/pandas/issues/22224
            try:
                subdtype = dt.subtype
            except AttributeError:
                continue
            else:
                dtypes.append(subdtype)
        else:
            dtypes.append(dt)

    dtype = np.result_type(*dtypes) if dtypes else np.float64
    # Create a meta object so that dask.array doesn't try to look
    # too closely at the type of the chunks it's wrapping
    # they're actually dataframes, tell dask they're ndarrays
    meta = np.empty((0,), dtype=dtype)
    # Create a chunks tuple, a singleton for each dataframe chunk
    # The number of chunks + structure needs to match that of
    # the dataframe, so that we can use the dataframe graph keys,
    # but we don't have to be precise with the chunk size.
    # We could use np.nan instead of 1 to indicate that we actually
    # don't know how large the chunk is
    chunks = (tuple(1 for _ in range(df.npartitions)),)

    # Now create a dask array from the dataframe graph layer
    # It's a dask array of dataframes, which is dodgy but useful
    # for the following reasons:
    #
    # (1) The dataframes get converted to a single array by
    #     the datashader reduction functions anyway
    # (2) dask.array.reduction is handy for coding a tree
    #     reduction of arrays
    df_array = da.Array(graph, df._name, chunks, meta=meta)
    # A sufficient condition for ensuring the chimera holds together
    assert list(df_array.__dask_keys__()) == list(df.__dask_keys__())

    def chunk(df, axis, keepdims):
        """ used in the dask.array.reduction chunk step """
        # df is a pandas.DataFrame computed from one dask.DataFrame partition
        aggs = create(shape)
        extend(aggs, df, st, bounds)
        return aggs

    def wrapped_combine(x, axis, keepdims):
        """ wrap datashader combine in dask.array.reduction combine """
        if isinstance(x, list):
            # list of tuples of ndarrays
            # assert all(isinstance(item, tuple) and
            #            len(item) == 1 and
            #            isinstance(item[0], np.ndarray)
            #            for item in x)
            return combine(x)
        elif isinstance(x, tuple):
            # tuple with single ndarray
            # assert len(x) == 1 and isinstance(x[0], np.ndarray)
            return x
        else:
            raise TypeError("Unknown type %s in wrapped_combine" % type(x))

    local_axis = axis

    def aggregate(x, axis, keepdims):
        """ Wrap datashader finalize in dask.array.reduction aggregate """
        return finalize(wrapped_combine(x, axis, keepdims),
                        cuda=cuda, coords=local_axis,
                        dims=[glyph.y_label, glyph.x_label],
                        attrs=dict(x_range=x_range, y_range=y_range))

    R = da.reduction(df_array,
                     aggregate=aggregate,
                     chunk=chunk,
                     combine=wrapped_combine,
                     # Control granularity of tree branching
                     # less is more
                     split_every=2,
                     # We don't want np.concatenate called
                     # during combine and aggregate. It'll
                     # fail because we're handling tuples of ndarrays
                     # and lists of tuples of ndarrays
                     concatenate=False,
                     # Prevent dask from internally inspecting
                     # chunk, combine and aggrregate
                     meta=meta,
                     # Provide some sort of dtype for the
                     # resultant dask array
                     dtype=meta.dtype)

    return R, R.name


@glyph_dispatch.register(LineAxis0)
def line(glyph, df, schema, canvas, summary, *, antialias=False, cuda=False):
    if cuda:
        from cudf import concat
    else:
        from pandas import concat

    shape, bounds, st, axis = shape_bounds_st_and_axis(df, canvas, glyph)

    # Compile functions
    df = _dask_compat(df)
    partitioned = isinstance(df, dd.DataFrame) and df.npartitions > 1
    create, info, append, combine, finalize, antialias_stage_2, antialias_stage_2_funcs, _ = \
        compile_components(summary, schema, glyph, antialias=antialias, cuda=cuda,
                           partitioned=partitioned)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(
        x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs)
    x_range = bounds[:2]
    y_range = bounds[2:]

    def chunk(df, df2=None):
        plot_start = True
        if df2 is not None:
            df = concat([df.iloc[-1:], df2])
            plot_start = False
        aggs = create(shape)
        extend(aggs, df, st, bounds, plot_start=plot_start)
        return aggs

    name = tokenize(df.__dask_tokenize__(), canvas, glyph, summary)
    old_name = df.__dask_tokenize__()
    # dask_expr return tokenize result as tuple of type and task name
    # We only want to use the task name as input to the new graph
    if isinstance(old_name, tuple):
        old_name = old_name[1]
    dsk = {(name, 0): (chunk, (old_name, 0))}
    for i in range(1, df.npartitions):
        dsk[(name, i)] = (chunk, (old_name, i - 1), (old_name, i))
    keys2 = [(name, i) for i in range(df.npartitions)]
    dsk[name] = (apply, finalize, [(combine, keys2)],
                 dict(cuda=cuda, coords=axis, dims=[glyph.y_label, glyph.x_label],
                      attrs=dict(x_range=x_range, y_range=y_range)))
    return dsk, name
