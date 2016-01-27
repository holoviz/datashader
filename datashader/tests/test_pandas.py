import numpy as np
import pandas as pd
from dynd import nd

import datashader as ds


df = pd.DataFrame({'x': np.array(([0.] * 10 + [1] * 10)),
                   'y': np.array(([0.] * 5 + [1] * 5 + [0] * 5 + [1] * 5)),
                   'log_x': np.array(([1.] * 10 + [10] * 10)),
                   'log_y': np.array(([1.] * 5 + [10] * 5 + [1] * 5 + [10] * 5)),
                   'i32': np.arange(20, dtype='i4'),
                   'i64': np.arange(20, dtype='i8'),
                   'f32': np.arange(20, dtype='f4'),
                   'f64': np.arange(20, dtype='f8'),
                   'cat': ['a']*5 + ['b']*5 + ['c']*5 + ['d']*5})
df.cat = df.cat.astype('category')

c = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))
c_logx = ds.Canvas(plot_width=2, plot_height=2, x_range=(1, 10),
                   y_range=(0, 1), x_axis_type='log')
c_logy = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1),
                   y_range=(1, 10), y_axis_type='log')
c_logxy = ds.Canvas(plot_width=2, plot_height=2, x_range=(1, 10),
                    y_range=(1, 10), x_axis_type='log', y_axis_type='log')


def eq(agg, b):
    agg = agg._data
    a = nd.as_numpy(agg.view_scalars(getattr(agg.dtype, 'value_type', agg.dtype)))
    assert np.allclose(a, b)
    assert a.dtype == b.dtype


def test_count():
    out = np.array([[5, 5], [5, 5]], dtype='i4')
    eq(c.points(df, 'x', 'y', ds.count('i32')), out)
    eq(c.points(df, 'x', 'y', ds.count('i64')), out)
    eq(c.points(df, 'x', 'y', ds.count('f32')), out)
    eq(c.points(df, 'x', 'y', ds.count('i64')), out)


def test_sum():
    out = df.i32.reshape((2, 2, 5)).sum(axis=2, dtype='i8').T
    eq(c.points(df, 'x', 'y', ds.sum('i32')), out)
    eq(c.points(df, 'x', 'y', ds.sum('i64')), out)
    out = out.astype('f8')
    eq(c.points(df, 'x', 'y', ds.sum('f32')), out)
    eq(c.points(df, 'x', 'y', ds.sum('f64')), out)


def test_min():
    out = df.i32.reshape((2, 2, 5)).min(axis=2).T
    eq(c.points(df, 'x', 'y', ds.min('i32')), out)
    eq(c.points(df, 'x', 'y', ds.min('i64')), out.astype('i8'))
    eq(c.points(df, 'x', 'y', ds.min('f32')), out.astype('f4'))
    eq(c.points(df, 'x', 'y', ds.min('f64')), out.astype('f8'))


def test_max():
    out = df.i32.reshape((2, 2, 5)).max(axis=2).T
    eq(c.points(df, 'x', 'y', ds.max('i32')), out)
    eq(c.points(df, 'x', 'y', ds.max('i64')), out.astype('i8'))
    eq(c.points(df, 'x', 'y', ds.max('f32')), out.astype('f4'))
    eq(c.points(df, 'x', 'y', ds.max('f64')), out.astype('f8'))


def test_mean():
    out = df.i32.reshape((2, 2, 5)).mean(axis=2).T
    eq(c.points(df, 'x', 'y', ds.mean('i32')), out)
    eq(c.points(df, 'x', 'y', ds.mean('i64')), out)
    eq(c.points(df, 'x', 'y', ds.mean('f32')), out)
    eq(c.points(df, 'x', 'y', ds.mean('f64')), out)


def test_var():
    out = df.i32.reshape((2, 2, 5)).var(axis=2).T
    eq(c.points(df, 'x', 'y', ds.var('i32')), out)
    eq(c.points(df, 'x', 'y', ds.var('i64')), out)
    eq(c.points(df, 'x', 'y', ds.var('f32')), out)
    eq(c.points(df, 'x', 'y', ds.var('f64')), out)


def test_std():
    out = df.i32.reshape((2, 2, 5)).std(axis=2).T
    eq(c.points(df, 'x', 'y', ds.std('i32')), out)
    eq(c.points(df, 'x', 'y', ds.std('i64')), out)
    eq(c.points(df, 'x', 'y', ds.std('f32')), out)
    eq(c.points(df, 'x', 'y', ds.std('f64')), out)


def test_count_cat():
    agg = c.points(df, 'x', 'y', ds.count_cat('cat'))
    sol = np.array([[[5, 0, 0, 0],
                     [0, 0, 5, 0]],
                    [[0, 5, 0, 0],
                     [0, 0, 0, 5]]])
    assert (nd.as_numpy(agg._data) == sol).all()
    assert agg._cats == ('a', 'b', 'c', 'd')


def test_multiple_aggregates():
    agg = c.points(df, 'x', 'y',
                   ds.summary(f64=ds.summary(std=ds.std('f64'),
                                             mean=ds.mean('f64')),
                              i32_sum=ds.sum('i32'),
                              i32_count=ds.count('i32')))

    eq(agg.f64.std, df.f64.reshape((2, 2, 5)).std(axis=2).T)
    eq(agg.f64.mean, df.f64.reshape((2, 2, 5)).mean(axis=2).T)
    eq(agg.i32_sum, df.i32.reshape((2, 2, 5)).sum(axis=2).T)
    eq(agg.i32_count, np.array([[5, 5], [5, 5]], dtype='i4'))


def test_log_axis():
    out = np.array([[5, 5], [5, 5]], dtype='i4')
    eq(c_logx.points(df, 'log_x', 'y', ds.count('i32')), out)
    eq(c_logy.points(df, 'x', 'log_y', ds.count('i32')), out)
    eq(c_logxy.points(df, 'log_x', 'log_y', ds.count('i32')), out)
