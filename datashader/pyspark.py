from __future__ import absolute_import, division

import pyspark.sql.functions as F

from pyspark.sql import DataFrame

from .core import bypixel
from .compiler import compile_components
from .glyphs import Glyph
from .utils import Dispatcher, pyspark_to_pandas_schema, serialized_rows_to_pandas


__all__ = ()


@bypixel.pipeline.register(DataFrame)
def pyspark_pipeline(df, schema, canvas, glyph, summary):
    return glyph_dispatch(glyph, df, schema, canvas, summary)


def shape_bounds_st_and_axis(df, canvas, glyph):

    data_bounds = df.select(F.min(glyph.x).alias("x_min"),
                            F.max(glyph.x).alias("x_max"),
                            F.min(glyph.y).alias("y_min"),
                            F.max(glyph.y).alias("y_max"))

    if (canvas.x_range is None) or (canvas.y_range is None):
        x_min, x_max, y_min, y_max = data_bounds.head()

    x_range = canvas.x_range or (x_min, x_max)
    y_range = canvas.y_range or (y_min, y_max)
    bounds = x_range + y_range

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
    """This is evaluated on any Glyph for any pyspark.sql.DataFrame"""

    # Determine shape and extent of data and plot
    # This is a mashup of the code I found in the existing pandas and dask methods
    shape, bounds, st, axis = shape_bounds_st_and_axis(df, canvas, glyph)
    x_min, x_max, y_min, y_max = bounds
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)
    x_st = canvas.x_axis.compute_scale_and_translate(x_range, canvas.plot_width)
    y_st = canvas.y_axis.compute_scale_and_translate(y_range, canvas.plot_height)
    x_axis = canvas.x_axis.compute_index(x_st, canvas.plot_width)
    y_axis = canvas.y_axis.compute_index(y_st, canvas.plot_height)

    # Compile functions
    create, info, append, combine, finalize = compile_components(summary, schema, glyph)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(x_mapper, y_mapper, info, append)

    # Use pyspark dataframe API to filter on and select only the columns we need to reduce the
    # serialization overhead. Select is especially helpful on wide dataframes, or those with
    # non-numeric data.
    columns = [glyph.x, glyph.y]
    if (summary.column is not None) and (summary.column not in columns):
        columns.append(summary.column)
    df = (df.filter(F.col(glyph.x).between(*x_range))
          .filter(F.col(glyph.y).between(*y_range))
          .select(*columns))
    schema = pyspark_to_pandas_schema(df.schema)

    # Map the aggregations to partitions of the dataframe
    parts = rdd_method(df, schema, shape, create, extend, st, bounds)

    # Reduce
    combined = combine(map(reversed, enumerate(parts)))[0]
    return finalize(combined, coords=[y_axis, x_axis], dims=[glyph.y, glyph.x])


def rdd_method(df, schema, shape, create, extend, st, bounds):
    """
    RDD-based serialization method for partial aggregation
    """
    # Read serialized rows in each partition as pandas dataframe, then delegate to datashader's
    # pandas methods
    def partition_fn(rows):
        df = serialized_rows_to_pandas(schema, rows)
        aggs = create(shape)
        extend(aggs, df, st, bounds)
        return [aggs]  # make sure to return wrapped in list so the results aren't flattened

    # Apply the function to each partition in the dataframe
    parts = df.rdd.mapPartitions(partition_fn).collect()
    return parts
