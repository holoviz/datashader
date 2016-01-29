from __future__ import division
import operator as op

from datashape import dshape
from dynd import nd
import numpy as np

from datashader.aggregates import (ScalarAggregate, CategoricalAggregate,
                                   RecordAggregate)
from datashader.core import LinearAxis, LogAxis

import pytest


x_axis = LinearAxis((0, 10))
y_axis = LinearAxis((1, 5))

a = nd.array([[0, 1, 2], [3, 4, None]], '2 * 3 * ?int64')
b = nd.array([[2, 2, None], [0, 3, 3]], '2 * 3 * ?float64')
c = nd.array([True, False, True])
d = nd.array([True, True, False])
s_a = ScalarAggregate(a, x_axis=x_axis, y_axis=y_axis)
s_b = ScalarAggregate(b, x_axis=x_axis, y_axis=y_axis)
s_c = ScalarAggregate(c, x_axis=x_axis, y_axis=y_axis)
s_d = ScalarAggregate(d, x_axis=x_axis, y_axis=y_axis)


def assert_dynd_eq(a, b, check_dtype=True):
    if check_dtype:
        assert a.dtype == b.dtype
    print(a)
    print(b)
    assert np.all((a == b).view_scalars('bool'))


def test_scalar_agg():
    assert s_a.dshape == dshape('?int64')
    assert s_b.dshape == dshape('?float64')
    assert s_a.shape == s_b.shape == (2, 3)


@pytest.mark.parametrize('op', [op.add, op.sub, op.mul, op.eq, op.ne,
                                op.lt, op.le, op.gt, op.ge])
def test_scalar_agg_binops(op):
    assert_dynd_eq(op(s_a, s_b)._data, op(a, b))
    assert_dynd_eq(op(s_a, 2)._data, op(a, 2))
    assert_dynd_eq(op(s_b, 2)._data, op(b, 2))
    assert_dynd_eq(op(s_a, 2.)._data, op(a, 2.))
    assert_dynd_eq(op(s_b, 2.)._data, op(b, 2.))
    assert_dynd_eq(op(2, s_a)._data, op(2, a))
    assert_dynd_eq(op(2, s_b)._data, op(2, b))
    assert_dynd_eq(op(2., s_a)._data, op(2., a))
    assert_dynd_eq(op(2., s_b)._data, op(2., b))


def test_binop_consistent_axis():
    # If/when non_aligned axis become supported, these can be removed
    with pytest.raises(NotImplementedError):
        s_a + ScalarAggregate(a, x_axis=x_axis, y_axis=LinearAxis((2, 5)))
    with pytest.raises(NotImplementedError):
        s_a + ScalarAggregate(a, x_axis=x_axis, y_axis=LogAxis((1, 5)))


def test_scalar_agg_truediv():
    assert_dynd_eq((s_a/s_b)._data, a/b)
    assert_dynd_eq((s_a/2)._data, a/2.)
    assert_dynd_eq((s_b/2)._data, b/2)
    assert_dynd_eq((s_a/2.)._data, a/2.)
    assert_dynd_eq((s_b/2.)._data, b/2.)
    assert_dynd_eq((2/s_a)._data, 2./a)
    assert_dynd_eq((2/s_b)._data, 2/b)
    assert_dynd_eq((2./s_a)._data, 2./a)
    assert_dynd_eq((2./s_b)._data, 2./b)


def test_scalar_agg_floordiv():
    def floordiv(a, b, out_type):
        o = np.floor((a/b).view_scalars('float64'))
        return nd.array(o, '2 * 3 * ' + out_type)
    assert_dynd_eq((s_a//s_b)._data, floordiv(a, b, '?float64'))
    assert_dynd_eq((s_a//2)._data, floordiv(a, 2., '?int64'))
    assert_dynd_eq((s_b//2)._data, floordiv(b, 2, '?float64'))
    assert_dynd_eq((s_a//2.)._data, floordiv(a, 2., '?float64'))
    assert_dynd_eq((s_b//2.)._data, floordiv(b, 2., '?float64'))
    assert_dynd_eq((2//s_a)._data, nd.array([[0, 2, 1], [0, 0, None]],
                                            '2 * 3 * ?int64'))
    assert_dynd_eq((2//s_b)._data, floordiv(2, b, '?float64'))
    assert_dynd_eq((2.//s_a)._data, nd.array([[nd.inf, 2, 1], [0, 0, None]]))
    assert_dynd_eq((2.//s_b)._data, floordiv(2., b, '?float64'))


@pytest.mark.parametrize('op', [op.and_, op.or_, op.xor])
def test_scalar_agg_bool(op):
    np_c = nd.as_numpy(c)
    np_d = nd.as_numpy(d)
    assert_dynd_eq(op(s_c, s_d)._data, op(np_c, np_d), False)
    assert_dynd_eq(op(s_c, True)._data, op(np_c, True), False)
    assert_dynd_eq(op(s_d, True)._data, op(np_d, True), False)
    assert_dynd_eq(op(s_c, True)._data, op(np_c, True), False)
    assert_dynd_eq(op(s_d, True)._data, op(np_d, True), False)
    assert_dynd_eq(op(True, s_c)._data, op(True, np_c), False)
    assert_dynd_eq(op(True, s_d)._data, op(True, np_d), False)
    assert_dynd_eq(op(True, s_c)._data, op(True, np_c), False)
    assert_dynd_eq(op(True, s_d)._data, op(True, np_d), False)


def test_scalar_agg_unops():
    assert_dynd_eq((+s_a)._data, a)
    assert_dynd_eq((+s_b)._data, b)
    assert_dynd_eq((-s_a)._data, -1 * a)
    assert_dynd_eq((-s_b)._data, -1 * b)
    # Hack around dynd's limited supported ops
    assert_dynd_eq((~s_a)._data, -1*a - 1)
    assert_dynd_eq(abs(s_a)._data, a)
    assert_dynd_eq(abs(s_b)._data, b)


def test_scalar_agg_where():
    assert_dynd_eq(s_a.where(s_a < 3)._data,
                   nd.array([[0, 1, 2], [None, None, None]], '2 * 3 * ?int64'))
    assert_dynd_eq(s_a.where(s_a < 3, 5)._data,
                   nd.array([[0, 1, 2], [5, 5, 5]], '2 * 3 * ?int64'))
    assert_dynd_eq(s_b.where(s_a < 3, 5)._data,
                   nd.array([[2, 2, None], [5, 5, 3]], '2 * 3 * ?float64'))
    assert_dynd_eq(s_b.where(s_a < 3, s_a)._data,
                   nd.array([[2, 2, None], [3, 4, 3]], '2 * 3 * ?float64'))
    with pytest.raises(TypeError):
        s_a.where(s_b)
    with pytest.raises(NotImplementedError):
        s_a.where(s_c)
    temp = ScalarAggregate((s_a > 2)._data, x_axis=x_axis,
                           y_axis=LogAxis((1, 5)))
    with pytest.raises(NotImplementedError):
        s_a.where(temp)


def test_categorical_agg():
    data = np.array([[(0, 12, 0), (3, 0, 3)],
                     [(12, 12, 12), (24, 0, 0)]], dtype='i4')
    cats = ['a', 'b', 'c']
    agg = CategoricalAggregate(nd.asarray(data), cats, x_axis, y_axis)
    assert agg.shape == (2, 2)
    assert agg.dshape == dshape('3 * int32')
    assert all(hasattr(agg, c) for c in cats)
    assert isinstance(agg['a'], ScalarAggregate)
    assert_dynd_eq(agg['a']._data, np.array([[0, 3], [12, 24]]), False)
    assert_dynd_eq(agg[['a', 'c']]._data, data[:, :, [0, 2]], False)
    with pytest.raises(KeyError):
        agg['d']
    with pytest.raises(KeyError):
        agg[['a', 'd']]
    with pytest.raises(AttributeError):
        agg.d


def test_record_agg():
    e = nd.array([[1, 2, 1.5], [3, 4, 5]], '2 * 3 * float64')
    s_e = ScalarAggregate(e, x_axis=x_axis, y_axis=y_axis)
    agg = RecordAggregate(dict(a=s_a, b=s_b, e=s_e), x_axis, y_axis)
    assert agg.shape == (2, 3)
    assert agg.dshape == dshape('{a: ?int64, b: ?float64, e: float64}')
    assert agg.keys() == ['a', 'b', 'e']
    assert all(hasattr(agg, c) for c in 'abe')
    assert isinstance(agg['a'], ScalarAggregate)
    sub = agg[['a', 'e']]
    assert isinstance(sub, RecordAggregate)
    assert hasattr(sub, 'a') and hasattr(sub, 'e')
    with pytest.raises(KeyError):
        agg['c']
    with pytest.raises(AttributeError):
        agg.c
    with pytest.raises(ValueError):
        agg = RecordAggregate(dict(a=s_a, b=s_b, c=s_c), x_axis, y_axis)
    temp = ScalarAggregate(e, x_axis=x_axis, y_axis=LogAxis((1, 5)))
    with pytest.raises(ValueError):
        agg = RecordAggregate(dict(a=s_a, b=s_b, temp=temp), x_axis, y_axis)
