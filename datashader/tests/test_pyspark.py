"""
Performs integration tests with Pandas implementations
"""
import findspark
import numpy as np
import pandas as pd
import pytest
import sys

from os.path import dirname, join
from pyspark.sql import SparkSession

import datashader as ds


# Start a sparkSession
spark_home = join(dirname(dirname(sys.executable)), 
                  "lib", 
                  "python{v.major}.{v.minor}".format(v=sys.version_info), 
                  "site-packages", 
                  "pyspark")
findspark.init(spark_home=spark_home, python_path=sys.executable)
spark = (SparkSession.builder
         .config("spark.master", "local[1]")
         .getOrCreate())
sc = spark.sparkContext



df = pd.DataFrame({'x': np.array(([0.] * 10 + [1] * 10)),
                   'y': np.array(([0.] * 5 + [1] * 5 + [0] * 5 + [1] * 5)),
                   'log_x': np.array(([1.] * 10 + [10] * 10)),
                   'log_y': np.array(([1.] * 5 + [10] * 5 + [1] * 5 + [10] * 5)),
                   'i32': np.arange(20, dtype='i4'),
                   'i64': np.arange(20, dtype='i8'),
                   'f32': np.arange(20, dtype='f4'),
                   'f64': np.arange(20, dtype='f8'),
                   'empty_bin': np.array([0.] * 15 + [np.nan] * 5),
                   'cat': ['a']*5 + ['b']*5 + ['c']*5 + ['d']*5})
df.cat = df.cat.astype('category')
df.f32[2] = np.nan
df.f64[2] = np.nan

sdf = spark.createDataFrame(df)
sdf.cache()

c = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))
c_logx = ds.Canvas(plot_width=2, plot_height=2, x_range=(1, 10),
                   y_range=(0, 1), x_axis_type='log')
c_logy = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1),
                   y_range=(1, 10), y_axis_type='log')
c_logxy = ds.Canvas(plot_width=2, plot_height=2, x_range=(1, 10),
                    y_range=(1, 10), x_axis_type='log', y_axis_type='log')

axis = ds.core.LinearAxis()
lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)
coords = [lincoords, lincoords]
dims = ['y', 'x']


canvi = {"c": c, "c_logx": c_logx, "c_logy": c_logy, "c_logxy": c_logxy}


@pytest.mark.parametrize("canvas", ["c", "c_logx", "c_logy", "c_logxy"])
@pytest.mark.parametrize("method", ["points", "line"])
@pytest.mark.parametrize("agg", ["count", "any", "sum", "min", "max", "mean", "var", "std"])
@pytest.mark.parametrize("z", ["i32", "i64", "f32", "f64"])
def test_results_equal_pandas(canvas, method, agg, z):
    """PySpark should return the same results as Pandas"""
    summary = getattr(ds, agg)(z)
    pandas_agg = getattr(canvi[canvas], method)(df, "x", "y", summary)
    pyspark_agg = getattr(canvi[canvas], method)(sdf, "x", "y", summary)
    assert pyspark_agg.equals(pandas_agg)


def test_multiple_aggregates():
    """...with multiple aggregates, too"""
    summary = ds.summary(f64_std=ds.std('f64'),
                         f64_mean=ds.mean('f64'),
                         i32_sum=ds.sum('i32'),
                         i32_count=ds.count('i32'))
    pandas_agg = c.points(df, 'x', 'y', summary)
    pyspark_agg = c.points(sdf, 'x', 'y', summary)
    assert pyspark_agg.equals(pandas_agg)
