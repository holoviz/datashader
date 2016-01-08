from __future__ import absolute_import, division, print_function

import numpy as np
from datashape import dshape, isnumeric, Record, Option, DataShape, maxtype
from datashape import coretypes as ct
from toolz import concat, unique, memoize, identity

from .utils import ngjit


# Dynd Missing Type Flags
_dynd_missing_types = {np.dtype('i2'): np.iinfo('i2').min,
                       np.dtype('i4'): np.iinfo('i4').min,
                       np.dtype('i8'): np.iinfo('i8').min,
                       np.dtype('f4'): np.nan,
                       np.dtype('f8'): np.nan}


def make_is_missing(m):
    return ngjit(lambda x: x == m)

# Lookup from dtype to function that checks if value is missing
_dynd_is_missing = {}
for dt, m in _dynd_missing_types.items():
    _dynd_is_missing[dt] = np.isnan if m is np.nan else make_is_missing(m)


def numpy_dtype(x):
    if hasattr(x, 'ty'):
        return numpy_dtype(x.ty)
    return x.to_numpy_dtype()


def optionify(d):
    if isinstance(d, DataShape):
        return DataShape(*(optionify(i) for i in d.parameters))
    return d if isinstance(d, Option) else Option(d)


class Aggregation(object):
    def __hash__(self):
        return hash((type(self), self.inputs))

    def __eq__(self, other):
        return type(self) is type(other) and self.inputs == other.inputs


class Reduction(Aggregation):
    def __init__(self, column):
        self.column = column

    def validate(self, in_dshape):
        if not isnumeric(in_dshape.measure[self.column]):
            raise ValueError("input must be numeric")

    def out_dshape(self, in_dshape):
        if hasattr(self, '_dshape'):
            return self._dshape
        return dshape(optionify(in_dshape.measure[self.column]))

    @property
    def inputs(self):
        return (self.column,)

    @property
    def _bases(self):
        return (self,)

    @property
    def _temps(self):
        return ()

    @memoize
    def _build_create(self, dshape):
        dtype = numpy_dtype(dshape.measure)
        value = _dynd_missing_types[dtype]
        return lambda shape: np.full(shape, value, dtype=dtype)


class count(Reduction):
    _dshape = dshape(Option(ct.int32))

    def validate(self, in_dshape):
        pass

    @memoize
    def _build_create(self, dshape):
        dtype = numpy_dtype(dshape.measure)
        return lambda shape: np.zeros(shape, dtype=dtype)

    def _build_append(self, dshape):
        return append_count

    def _build_combine(self, dshape):
        return combine_count

    def _build_finalize(self, dshape):
        return identity


class sum(Reduction):
    def out_dshape(self, input_dshape):
        return dshape(optionify(maxtype(input_dshape.measure[self.column])))

    def _build_append(self, dshape):
        return build_append_sum(dshape)

    def _build_combine(self, dshape):
        return combine_sum

    def _build_finalize(self, dshape):
        return identity


class min(Reduction):
    @memoize
    def _build_create(self, dshape):
        dtype = numpy_dtype(dshape.measure)
        if np.issubdtype(dtype, np.floating):
            value = np.inf
        else:
            value = np.iinfo(dtype).max
        return lambda shape: np.full(shape, value, dtype=dtype)

    def _build_append(self, dshape):
        return append_min

    def _build_combine(self, dshape):
        return combine_min

    def _build_finalize(self, dshape):
        return build_finalize_min(dshape.measure[self.column])


class max(Reduction):
    @memoize
    def _build_create(self, dshape):
        dtype = numpy_dtype(dshape.measure)
        if np.issubdtype(dtype, np.floating):
            value = -np.inf
        else:
            value = np.iinfo(dtype).min
        return lambda shape: np.full(shape, value, dtype=dtype)

    def _build_append(self, dshape):
        return append_max

    def _build_combine(self, dshape):
        return combine_max

    def _build_finalize(self, dshape):
        return build_finalize_max(dshape.measure[self.column])


class m2(Reduction):
    """Second moment"""
    _dshape = dshape(ct.float64)

    @property
    def _temps(self):
        return (sum(self.column), count(self.column))

    def _build_append(self, dshape):
        return append_m2

    def _build_combine(self, dshape):
        return combine_m2

    def _build_finalize(self, dshape):
        return identity


class mean(Reduction):
    _dshape = dshape(Option(ct.float64))

    @property
    def _bases(self):
        return (sum(self.column), count(self.column))

    def _build_finalize(self, dshape):
        return finalize_mean


class var(Reduction):
    _dshape = dshape(Option(ct.float64))

    @property
    def _bases(self):
        return (sum(self.column), count(self.column), m2(self.column))

    def _build_finalize(self, dshape):
        return finalize_var


class std(Reduction):
    _dshape = dshape(Option(ct.float64))

    @property
    def _bases(self):
        return (sum(self.column), count(self.column), m2(self.column))

    def _build_finalize(self, dshape):
        return finalize_std


class Summary(Aggregation):
    def __init__(self, **kwargs):
        ks, vs = zip(*sorted(kwargs.items()))
        self.keys = ks
        self.values = [Summary(**v) if isinstance(v, dict) else v for v in vs]

    def __hash__(self):
        return hash((type(self), tuple(self.keys), tuple(self.values)))

    def validate(self, input_dshape):
        for v in self.values:
            v.validate(input_dshape)

    def out_dshape(self, in_dshape):
        return dshape(Record([(k, v.out_dshape(in_dshape)) for (k, v)
                              in zip(self.keys, self.values)]))

    @property
    def inputs(self):
        return tuple(unique(concat(v.inputs for v in self.values)))


# ============ Appenders ============

@ngjit
def append_count(x, y, agg, field):
    agg[y, x] += 1


@ngjit
def append_max(x, y, agg, field):
    if agg[y, x] < field:
        agg[y, x] = field


@ngjit
def append_min(x, y, agg, field):
    if agg[y, x] > field:
        agg[y, x] = field


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


@memoize
def build_append_sum(dshape):
    # sum needs specialization for each missing flag
    dtype = numpy_dtype(dshape)
    is_missing = _dynd_is_missing[dtype]

    @ngjit
    def append_sum(x, y, agg, field):
        if is_missing(agg[y, x]):
            agg[y, x] = field
        else:
            agg[y, x] += field
    return append_sum


# ============ Combiners ============

def combine_count(aggs):
    return aggs.sum(axis=0)


def combine_sum(aggs):
    missing_val = _dynd_missing_types[aggs.dtype]
    is_missing = _dynd_is_missing[aggs.dtype]
    missing_vals = is_missing(aggs)
    all_empty = np.bitwise_and.reduce(missing_vals, axis=0)
    set_to_zero = missing_vals & ~all_empty
    out = np.where(set_to_zero, 0, aggs).sum(axis=0)
    if missing_val is not np.nan:
        out[all_empty] = missing_val
    return out


def combine_min(aggs):
    return np.nanmin(aggs, axis=0)


def combine_max(aggs):
    return np.nanmax(aggs, axis=0)


def combine_m2(Ms, sums, ns):
    sums = as_float64(sums)
    mu = np.nansum(sums, axis=0) / ns.sum(axis=0)
    return np.nansum(Ms + ns*(sums/ns - mu)**2, axis=0)


# ============ Finalizers ============

@memoize
def build_finalize_min(dshape):
    dtype = numpy_dtype(dshape.measure)
    missing = _dynd_missing_types[dtype]
    if np.issubdtype(dtype, np.floating):
        return lambda x: np.where(np.isposinf(x), missing, x)
    else:
        value = np.iinfo(dtype).max
        return lambda x: np.where(x == value, missing, x)


@memoize
def build_finalize_max(dshape):
    dtype = numpy_dtype(dshape.measure)
    missing = _dynd_missing_types[dtype]
    if np.issubdtype(dtype, np.floating):
        return lambda x: np.where(np.isneginf(x), missing, x)
    else:
        value = np.iinfo(dtype).min
        return lambda x: np.where(x == value, missing, x)


def as_float64(arr):
    is_missing = _dynd_is_missing[arr.dtype]
    return np.where(is_missing(arr), np.nan, arr.astype('f8'))


def finalize_mean(sums, counts):
    with np.errstate(divide='ignore', invalid='ignore'):
        return as_float64(sums)/counts


def finalize_var(sums, counts, m2s):
    with np.errstate(divide='ignore', invalid='ignore'):
        return as_float64(m2s)/counts


def finalize_std(sums, counts, m2s):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(as_float64(m2s)/counts)
