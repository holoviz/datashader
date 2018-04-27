"""
Datashader methods and helper functions to bootstrap extant pandas methods onto pyspark dataframes

Notes
-----
* Since Spark has a .collect in .reduce, full reduction by datashader is probably faster than binary
  reduction in a .reduce call
* I have avoided .reduceByKey because it just collects everything onto a single executor
* I have avoided .toLocalIterator because it single-threads the computation
"""

from __future__ import absolute_import, division

import datashape
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T

from collections import OrderedDict
from pyspark.sql import DataFrame

from datashader.compatibility import zip
from datashader.core import LinearAxis, LogAxis, bypixel
from datashader.compiler import compile_components
from datashader.glyphs import Glyph, Line
from datashader.utils import Dispatcher, dshape_from_pandas, necessary_columns

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


def prep(glyph, df, schema, canvas, summary):
    """Shared preparation for glyph methods"""
    
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
    if isinstance(canvas.x_axis, LinearAxis):             # Not able to reproduce pandas results
        df = df.filter(F.col(glyph.x).between(*x_range))  # with LogAxis and non-count based
    if isinstance(canvas.y_axis, LinearAxis):             # reductions. The filter I want to do is
        df = df.filter(F.col(glyph.y).between(*y_range))  # sql_project(canvas.{x|y}_axis, glyph.{x|y})
                                                          #   .between(*map({x|y}_mapper, {x|y}_range))
    columns = necessary_columns(glyph, summary)
    df = df.select(*columns)
    pandas_schema, nullsafe_schema = pyspark_to_pandas_schema(df.schema)

    # # Handle the count_cat summary case
    # if isinstance(summary, count_cat):
    #     categories = sorted(getattr(row, summary.column) for row in
    #                         df.select(summary.column).distinct().collect())
    #     pandas_schema[summary.column] = CategoricalDtype(categories)
    #     summary._add_categories(categories)
    
    return create, info, append, combine, finalize, extend, shape, st, bounds, y_axis, x_axis, \
           pandas_schema, nullsafe_schema, df


@glyph_dispatch.register(Glyph)
def default(glyph, df, schema, canvas, summary):
    """This is evaluated on any Glyph for any pyspark.sql.DataFrame"""

    create, info, append, combine, finalize, extend, shape, st, bounds, y_axis, x_axis, \
        pandas_schema, nullsafe_schema, df = prep(glyph, df, schema, canvas, summary)

    # Read serialized rows in each partition as pandas dataframe, then delegate to datashader's
    # pandas methods
    def partition_fn(rows):
        df = serialized_rows_to_pandas(pandas_schema, nullsafe_schema, rows)
        aggs = create(shape)
        extend(aggs, df, st, bounds)
        return [aggs]  # make sure to return wrapped in list so the results aren't flattened

    # Apply the function to each partition in the dataframe and collect the results
    parts = df.rdd.mapPartitions(partition_fn).collect()
    # result = combine(map(reversed, enumerate(parts)))[0]
    result = combine(parts)

    return finalize(result, coords=[y_axis, x_axis], dims=[glyph.y, glyph.x])


@glyph_dispatch.register(Line)
def line(glyph, df, schema, canvas, summary):
    
    create, info, append, combine, finalize, extend, shape, st, bounds, y_axis, x_axis, \
        pandas_schema, nullsafe_schema, df = prep(glyph, df, schema, canvas, summary)

    # Read serialized rows in each partition as pandas dataframe, then delegate to datashader's
    # pandas line method
    def indexed_partition_fn(index, rows):
        plot_start = (index == 0)
        df = serialized_rows_to_pandas(pandas_schema, nullsafe_schema, rows)
        aggs = create(shape)
        extend(aggs, df, st, bounds, plot_start=plot_start)
        return [aggs]  # make sure to return wrapped in list so the results aren't flattened

    # Apply the function to each partition in the dataframe and collect the results
    parts = df.rdd.mapPartitionsWithIndex(indexed_partition_fn).collect()
    # result = combine(map(reversed, enumerate(parts)))[0]
    result = combine(parts)

    return finalize(result, coords=[y_axis, x_axis], dims=[glyph.y, glyph.x])


TYPE_MAPPING = {
    T.ArrayType: np.object,
    T.BooleanType: np.bool,
    T.ByteType: np.int8,
    T.DecimalType: np.float64,
    T.DoubleType: np.float64,
    T.FloatType: np.float32,
    T.IntegerType: np.int32,
    T.LongType: np.int64,
    T.MapType: np.object,
    T.NullType: np.object,
    T.ShortType: np.int16,
    T.StructType: np.object,
    T.StringType: np.object,
    T.TimestampType: np.datetime64,
}


def pyspark_type_to_pandas_type(data_type):
    """
    Map a PySpark type to a pandas (numpy) type
    """
    for pyspark_type, pandas_type in TYPE_MAPPING.items():
        if isinstance(data_type, pyspark_type):
            return pandas_type
    raise TypeError("Unrecognized PySpark type %r" % data_type)


def nullsafe_pandas_type(numpy_type):
    """
    Map a pandas (numpy) type to its nullsafe equivalent
    """
    # objects and floating point naturally support null values
    if numpy_type in (np.object, np.float32, np.float64):
        return numpy_type

    # all other types can be safely captured by object **when used in pandas dataframes**
    return np.object


def pyspark_to_pandas_schema(pyspark_schema):
    """
    Map a pyspark schema to a pandas schema of {column name: type} and {column name: nullsafe type}
    """
    pandas_schema, nullsafe_schema = OrderedDict(), OrderedDict()
    for field in pyspark_schema:
        pandas_type = pyspark_type_to_pandas_type(field.dataType)
        pandas_schema[field.name] = pandas_type
        nullsafe_schema[field.name] = nullsafe_pandas_type(pandas_type) if field.nullable \
            else pandas_type
    return pandas_schema, nullsafe_schema


def empty_typed_dataframe(schema):
    """
    Create an empty, typed dataframe from a pandas schema
    """
    return pd.DataFrame({name: np.empty(0, dtype=dtype) for name, dtype in schema.items()})


def serialized_rows_to_pandas(pandas_schema, nullsafe_schema, rows):
    """
    Read a serialized collection of rows (an iterator of pyspark.sql.Row objects) as a pandas
    dataframe. Used to create pandas dataframes on executors in `rdd_method`'s mapping operation.
    """
    # Nullsafe conversion to dataframe
    df = pd.DataFrame.from_dict({name: np.array(values, dtype=dtype) for (name, dtype), values in 
                                 zip(nullsafe_schema.items(), zip(*rows))})

    # If the iterator was empty, df will have no rows or columns
    # We don't want to return that--instead return zero-row, typed dataframe
    if len(df) == 0:
        return empty_typed_dataframe(pandas_schema)

    # Converts columns to their intended types
    # This will raise an error if there are nulls in a type that can't handle them. This is desired
    # behavior: force the user to explicitly deal with nulls.
    return df.astype(pandas_schema)


def dshape_from_pyspark(df):
    """
    Return a datashape.DataShape object given a pyspark dataframe
    """
    schema, _ = pyspark_to_pandas_schema(df.schema)
    empty = empty_typed_dataframe(schema)
    return datashape.var * dshape_from_pandas(empty).measure


def sql_project(axis, column):
    """
    Project a column into the correct space based on the axis
    """
    if isinstance(axis, LogAxis):
        return F.log10(column)
    else:
        return F.col(column)
