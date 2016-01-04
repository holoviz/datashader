from __future__ import division

import numpy as np
from toolz import memoize
from blaze import dispatch
from blaze.expr import Field
from blaze.expr.reductions import (count, sum, mean, min, max, var, std,
                                   Reduction, FloatingReduction)

from .util import ngjit

__all__ = ['get_bases', 'get_create', 'get_cols', 'get_info', 'get_temps',
           'get_append', 'get_finalize']


class m2(FloatingReduction):
    """Second moment"""
    pass


@dispatch((sum, min, max, count))
def get_bases(red):
    """Return a tuple of the base types needed for this reduction"""
    return (red,)


@dispatch(mean)
def get_bases(red):
    return (sum(red._child), count(red._child))


@dispatch((std, var))
def get_bases(red):
    return (sum(red._child), count(red._child), m2(red._child))


# Dynd Missing Type Flags
_dynd_missing_types = {np.dtype('i2'): np.iinfo('i2').min,
                       np.dtype('i4'): np.iinfo('i4').min,
                       np.dtype('i8'): np.iinfo('i8').min,
                       np.dtype('f4'): np.nan,
                       np.dtype('f8'): np.nan}

_dynd_is_missing = {}
mk_missing = lambda m: ngjit(lambda x: x == m)
for dt, m in _dynd_missing_types.items():
    _dynd_is_missing[dt] = np.isnan if m is np.nan else mk_missing(m)


def numpy_dtype(x):
    if hasattr(x, 'ty'):
        return numpy_dtype(x.ty)
    return x.to_numpy_dtype()


@dispatch(Reduction)
def get_create(red):
    dtype = numpy_dtype(red.dshape.measure)
    value = _dynd_missing_types[dtype]
    return lambda shape: np.full(shape, value, dtype=dtype)


@dispatch(count)
def get_create(red):
    dtype = numpy_dtype(red.dshape.measure)
    return lambda shape: np.zeros(shape, dtype=dtype)


@dispatch(min)
def get_create(red):
    dtype = numpy_dtype(red.dshape.measure)
    if np.issubdtype(dtype, np.floating):
        value = np.inf
    else:
        value = np.iinfo(dtype).max
    return lambda shape: np.full(shape, value, dtype=dtype)


@dispatch(max)
def get_create(red):
    dtype = numpy_dtype(red.dshape.measure)
    if np.issubdtype(dtype, np.floating):
        value = -np.inf
    else:
        value = np.iinfo(dtype).min
    return lambda shape: np.full(shape, value, dtype=dtype)


@dispatch(Reduction)
def get_cols(red):
    return (red._child,)


@dispatch(Field)
def get_info(x):
    name = x._name
    return lambda df: df[name].values


@dispatch(Reduction)
def get_temps(red):
    return ()


@dispatch(m2)
def get_temps(red):
    return (sum(red._child), count(red._child))


@dispatch(Reduction)
def get_append(red):
    raise TypeError("Don't know how to handle {0}".format(red))


def register_append(red):
    def _(f):
        get_append.add((red,), lambda x: f)
        return f
    return _


@register_append(count)
@ngjit
def append_count(x, y, agg, field):
    agg[y, x] += 1


@dispatch(sum)
@memoize
def get_append(red):
    dtype = numpy_dtype(red.dshape.measure)
    is_missing = _dynd_is_missing[dtype]
    @ngjit
    def append_sum(x, y, agg, field):
        if is_missing(agg[y, x]):
            agg[y, x] = field
        else:
            agg[y, x] += field
    return append_sum


@register_append(max)
@ngjit
def append_max(x, y, agg, field):
    if agg[y, x] < field:
        agg[y, x] = field


@register_append(min)
@ngjit
def append_min(x, y, agg, field):
    if agg[y, x] > field:
        agg[y, x] = field


@register_append(m2)
@ngjit
def append_m2(x, y, m2, field, sum, count):
    # sum & count are the results of sum[y, x], count[y, x] before being
    # updated by field
    if count == 0:
        m2[y, x] = 0
    else:
        u1 = np.float64(sum) / count
        u = np.float64(sum + field) / (count + 1)
        m2[y, x] += (field - u1) * (field - u)


@dispatch((count, sum))
def get_finalize(red):
    return lambda x: x


@dispatch(min)
def get_finalize(red):
    dtype = numpy_dtype(red.dshape.measure)
    missing = _dynd_missing_types[dtype]
    if np.issubdtype(dtype, np.floating):
        return lambda x: np.where(np.isposinf(x), missing, x)
    else:
        value = np.iinfo(dtype).max
        return lambda x: np.where(x == value, missing, x)


@dispatch(max)
def get_finalize(red):
    dtype = numpy_dtype(red.dshape.measure)
    missing = _dynd_missing_types[dtype]
    if np.issubdtype(dtype, np.floating):
        return lambda x: np.where(np.isneginf(x), missing, x)
    else:
        value = np.iinfo(dtype).min
        return lambda x: np.where(x == value, missing, x)


def register_finalize(red):
    def _(f):
        get_finalize.add((red,), lambda x: f)
        return f
    return _


def as_float64(arr):
    is_missing = _dynd_is_missing[arr.dtype]
    return np.where(is_missing(arr), np.nan, arr.astype('f8'))


@register_finalize(mean)
def finalize_mean(sums, counts):
    with np.errstate(divide='ignore', invalid='ignore'):
        return as_float64(sums)/counts


@register_finalize(var)
def finalize_var(sums, counts, m2s):
    with np.errstate(divide='ignore', invalid='ignore'):
        return as_float64(m2s)/counts


@register_finalize(std)
def finalize_std(sums, counts, m2s):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(as_float64(m2s)/counts)
