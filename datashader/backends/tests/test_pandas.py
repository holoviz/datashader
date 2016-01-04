import numpy as np
import pandas as pd
from blaze import TableSymbol, compute, discover, summary
from dynd import nd

from datashader import Canvas


data = pd.DataFrame({'x': np.array(([0.] * 10 + [1] * 10)),
                     'y': np.array(([0.] * 5 + [1] * 5 + [0] * 5 + [1] * 5)),
                     'i32': np.arange(20, dtype='i4'),
                     'i64': np.arange(20, dtype='i8'),
                     'f32': np.arange(20, dtype='f4'),
                     'f64': np.arange(20, dtype='f8')})

df = TableSymbol('df', discover(data))

c = Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))


def eq(agg, b):
    arr = compute(agg, data).agg
    assert str(agg.agg.schema) == str(arr.dtype)
    a = nd.as_numpy(arr.view_scalars(arr.dtype.value_type))
    assert np.allclose(a, b)
    assert a.dtype == b.dtype


def test_count():
    out = np.array([[5, 5], [5, 5]], dtype='i4')
    eq(c.points(df.x, df.y, agg=df.i32.count()), out)
    eq(c.points(df.x, df.y, agg=df.i64.count()), out)
    eq(c.points(df.x, df.y, agg=df.f32.count()), out)
    eq(c.points(df.x, df.y, agg=df.f64.count()), out)


def test_sum():
    out = data.i32.reshape((2, 2, 5)).sum(axis=2, dtype='i8').T
    eq(c.points(df.x, df.y, agg=df.i32.sum()), out)
    eq(c.points(df.x, df.y, agg=df.i64.sum()), out)
    out = out.astype('f8')
    eq(c.points(df.x, df.y, agg=df.f32.sum()), out)
    eq(c.points(df.x, df.y, agg=df.f64.sum()), out)


def test_min():
    out = data.i32.reshape((2, 2, 5)).min(axis=2).T
    eq(c.points(df.x, df.y, agg=df.i32.min()), out)
    eq(c.points(df.x, df.y, agg=df.i64.min()), out.astype('i8'))
    eq(c.points(df.x, df.y, agg=df.f32.min()), out.astype('f4'))
    eq(c.points(df.x, df.y, agg=df.f64.min()), out.astype('f8'))


def test_max():
    out = data.i32.reshape((2, 2, 5)).max(axis=2).T
    eq(c.points(df.x, df.y, agg=df.i32.max()), out)
    eq(c.points(df.x, df.y, agg=df.i64.max()), out.astype('i8'))
    eq(c.points(df.x, df.y, agg=df.f32.max()), out.astype('f4'))
    eq(c.points(df.x, df.y, agg=df.f64.max()), out.astype('f8'))


def test_mean():
    out = data.i32.reshape((2, 2, 5)).mean(axis=2).T
    eq(c.points(df.x, df.y, agg=df.i32.mean()), out)
    eq(c.points(df.x, df.y, agg=df.i64.mean()), out)
    eq(c.points(df.x, df.y, agg=df.f32.mean()), out)
    eq(c.points(df.x, df.y, agg=df.f64.mean()), out)


def test_var():
    out = data.i32.reshape((2, 2, 5)).var(axis=2).T
    eq(c.points(df.x, df.y, agg=df.i32.var()), out)
    eq(c.points(df.x, df.y, agg=df.i64.var()), out)
    eq(c.points(df.x, df.y, agg=df.f32.var()), out)
    eq(c.points(df.x, df.y, agg=df.f64.var()), out)


def test_std():
    out = data.i32.reshape((2, 2, 5)).std(axis=2).T
    eq(c.points(df.x, df.y, agg=df.i32.std()), out)
    eq(c.points(df.x, df.y, agg=df.i64.std()), out)
    eq(c.points(df.x, df.y, agg=df.f32.std()), out)
    eq(c.points(df.x, df.y, agg=df.f64.std()), out)


def test_multiple_aggregates():
    agg = c.points(df.x, df.y,
                   f64=summary(std=df.f64.std(), mean=df.f64.mean()),
                   i32_sum=df.i32.sum(),
                   i32_count=df.i32.count())
    arr = compute(agg, data)
    asnp = lambda arr: nd.as_numpy(arr.view_scalars(arr.dtype.value_type))
    assert str(agg.schema) == str(arr.dtype)
    assert np.allclose(asnp(arr.f64.std),
                       data.f64.reshape((2, 2, 5)).std(axis=2).T)
    assert np.allclose(asnp(arr.f64.mean),
                       data.f64.reshape((2, 2, 5)).mean(axis=2).T)
    assert np.allclose(asnp(arr.i32_sum),
                       data.i32.reshape((2, 2, 5)).sum(axis=2).T)
    assert np.allclose(asnp(arr.i32_count),
                       np.array([[5, 5], [5, 5]], dtype='i4'))
