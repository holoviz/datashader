import numpy as np
import pandas as pd
from dynd import nd
import dask.dataframe as dd
from dask.context import set_options
from dask.async import get_sync
set_options(get=get_sync)

import datashader as ds


df = pd.DataFrame({'x': np.array(([0.] * 10 + [1] * 10)),
                   'y': np.array(([0.] * 5 + [1] * 5 + [0] * 5 + [1] * 5)),
                   'i32': np.arange(20, dtype='i4'),
                   'i64': np.arange(20, dtype='i8'),
                   'f32': np.arange(20, dtype='f4'),
                   'f64': np.arange(20, dtype='f8')})

ddf = dd.from_pandas(df, npartitions=3)

c = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))


def eq(agg, b):
    a = nd.as_numpy(agg.view_scalars(agg.dtype.value_type))
    assert np.allclose(a, b)
    assert a.dtype == b.dtype


def test_count():
    out = np.array([[5, 5], [5, 5]], dtype='i4')
    eq(c.points(ddf, 'x', 'y', agg=ds.count('i32')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.count('i64')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.count('f32')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.count('i64')).agg, out)


def test_sum():
    out = df.i32.reshape((2, 2, 5)).sum(axis=2, dtype='i8').T
    eq(c.points(ddf, 'x', 'y', agg=ds.sum('i32')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.sum('i64')).agg, out)
    out = out.astype('f8')
    eq(c.points(ddf, 'x', 'y', agg=ds.sum('f32')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.sum('f64')).agg, out)


def test_min():
    out = df.i32.reshape((2, 2, 5)).min(axis=2).T
    eq(c.points(ddf, 'x', 'y', agg=ds.min('i32')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.min('i64')).agg, out.astype('i8'))
    eq(c.points(ddf, 'x', 'y', agg=ds.min('f32')).agg, out.astype('f4'))
    eq(c.points(ddf, 'x', 'y', agg=ds.min('f64')).agg, out.astype('f8'))


def test_max():
    out = df.i32.reshape((2, 2, 5)).max(axis=2).T
    eq(c.points(ddf, 'x', 'y', agg=ds.max('i32')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.max('i64')).agg, out.astype('i8'))
    eq(c.points(ddf, 'x', 'y', agg=ds.max('f32')).agg, out.astype('f4'))
    eq(c.points(ddf, 'x', 'y', agg=ds.max('f64')).agg, out.astype('f8'))


def test_mean():
    out = df.i32.reshape((2, 2, 5)).mean(axis=2).T
    eq(c.points(ddf, 'x', 'y', agg=ds.mean('i32')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.mean('i64')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.mean('f32')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.mean('f64')).agg, out)


def test_var():
    out = df.i32.reshape((2, 2, 5)).var(axis=2).T
    eq(c.points(ddf, 'x', 'y', agg=ds.var('i32')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.var('i64')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.var('f32')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.var('f64')).agg, out)


def test_std():
    out = df.i32.reshape((2, 2, 5)).std(axis=2).T
    eq(c.points(ddf, 'x', 'y', agg=ds.std('i32')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.std('i64')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.std('f32')).agg, out)
    eq(c.points(ddf, 'x', 'y', agg=ds.std('f64')).agg, out)


def test_multiple_aggregates():
    agg = c.points(ddf, 'x', 'y',
                   f64=dict(std=ds.std('f64'), mean=ds.mean('f64')),
                   i32_sum=ds.sum('i32'),
                   i32_count=ds.count('i32'))

    eq(agg.f64.std, df.f64.reshape((2, 2, 5)).std(axis=2).T)
    eq(agg.f64.mean, df.f64.reshape((2, 2, 5)).mean(axis=2).T)
    eq(agg.i32_sum, df.i32.reshape((2, 2, 5)).sum(axis=2).T)
    eq(agg.i32_count, np.array([[5, 5], [5, 5]], dtype='i4'))
