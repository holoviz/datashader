from __future__ import absolute_import, division, print_function

import numpy as np
from dynd import nd
from datashape import dshape, isnumeric, Record, Option, DataShape, maxtype
from datashape import coretypes as ct
from toolz import concat, unique, memoize

from .aggregates import ScalarAggregate, ByCategoriesAggregate
from .utils import ngjit, is_missing


# Dynd Missing Type Flags
_dynd_missing_types = {np.dtype('i2'): np.iinfo('i2').min,
                       np.dtype('i4'): np.iinfo('i4').min,
                       np.dtype('i8'): np.iinfo('i8').min,
                       np.dtype('f4'): np.nan,
                       np.dtype('f8'): np.nan}


def numpy_dtype(x):
    if hasattr(x, 'ty'):
        return numpy_dtype(x.ty)
    return x.to_numpy_dtype()


def optionify(d):
    if isinstance(d, DataShape):
        return DataShape(*(optionify(i) for i in d.parameters))
    return d if isinstance(d, Option) else Option(d)


class Expr(object):
    def __init__(self, column):
        self.column = column

    def __hash__(self):
        return hash((type(self), self.inputs))

    def __eq__(self, other):
        return type(self) is type(other) and self.inputs == other.inputs


class Preprocess(Expr):
    def __init__(self, column):
        self.column = column

    @property
    def inputs(self):
        return (self.column,)


class extract(Preprocess):
    def apply(self, df):
        return df[self.column].values


class category_codes(Preprocess):
    def apply(self, df):
        return df[self.column].cat.codes.values


class Reduction(Expr):
    def validate(self, in_dshape):
        if not isnumeric(in_dshape.measure[self.column]):
            raise ValueError("input must be numeric")

    def out_dshape(self, in_dshape):
        if hasattr(self, '_dshape'):
            return self._dshape
        return dshape(optionify(in_dshape.measure[self.column]))

    @property
    def inputs(self):
        return (extract(self.column),)

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
    _dshape = dshape(ct.int32)

    def validate(self, in_dshape):
        pass

    @memoize
    def _build_create(self, dshape):
        return lambda shape: np.zeros(shape, dtype='i4')

    def _build_append(self, dshape):
        return append_count

    def _build_combine(self, dshape):
        return combine_count

    def _build_finalize(self, dshape):
        return build_finalize_identity(ct.int32)


class sum(Reduction):
    def out_dshape(self, input_dshape):
        return dshape(optionify(maxtype(input_dshape.measure[self.column])))

    def _build_append(self, dshape):
        return append_sum

    def _build_combine(self, dshape):
        return combine_sum

    def _build_finalize(self, dshape):
        return build_finalize_identity(self.out_dshape(dshape))


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
        return build_finalize_identity(ct.float64)


class count_cat(Reduction):
    def validate(self, in_dshape):
        if not isinstance(in_dshape.measure[self.column], ct.Categorical):
            raise ValueError("input must be categorical")

    def out_dshape(self, input_dshape):
        cats = input_dshape.measure[self.column].categories
        return dshape(Record([(c, ct.int32) for c in cats]))

    @property
    def inputs(self):
        return (category_codes(self.column),)

    def _build_create(self, out_dshape):
        n_cats = len(out_dshape.measure.fields)
        return lambda shape: np.zeros(shape + (n_cats,), dtype='i4')

    def _build_append(self, dshape):
        return append_count_cat

    def _build_combine(self, dshape):
        return combine_count_cat

    def _build_finalize(self, dshape):
        cats = dshape[self.column].categories
        def finalize(bases, **kwargs):
            return ByCategoriesAggregate(nd.asarray(bases[0]), cats, **kwargs)
        return finalize


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


class summary(Expr):
    def __init__(self, **kwargs):
        ks, vs = zip(*sorted(kwargs.items()))
        self.keys = ks
        self.values = vs

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


@ngjit
def append_sum(x, y, agg, field):
    if is_missing(agg[y, x]):
        agg[y, x] = field
    else:
        agg[y, x] += field


@ngjit
def append_count_cat(x, y, agg, field):
    agg[y, x, field] += 1


# ============ Combiners ============

def combine_count(aggs):
    return aggs.sum(axis=0, dtype='i4')


def combine_sum(aggs):
    missing_val = _dynd_missing_types[aggs.dtype]
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


def combine_count_cat(aggs):
    return aggs.sum(axis=0, dtype='i4')


# ============ Finalizers ============

def build_finalize_identity(dshape):
    dshape = str(dshape)
    def finalize(bases, **kwargs):
        return ScalarAggregate(nd.asarray(bases[0]).view_scalars(dshape),
                               **kwargs)
    return finalize


@memoize
def build_finalize_min(dshape):
    dtype = numpy_dtype(dshape.measure)
    missing = _dynd_missing_types[dtype]
    if np.issubdtype(dtype, np.floating):
        is_missing = np.isposinf
    else:
        value = np.iinfo(dtype).max
        is_missing = lambda x: x == value
    dshape = str(dshape)
    def finalize(bases, **kwargs):
        x = np.where(is_missing(bases[0]), missing, bases[0])
        return ScalarAggregate(nd.asarray(x).view_scalars(dshape), **kwargs)
    return finalize


@memoize
def build_finalize_max(dshape):
    dtype = numpy_dtype(dshape.measure)
    missing = _dynd_missing_types[dtype]
    if np.issubdtype(dtype, np.floating):
        is_missing = np.isneginf
    else:
        value = np.iinfo(dtype).max
        is_missing = lambda x: x == value
    dshape = str(dshape)
    def finalize(bases, **kwargs):
        x = np.where(is_missing(bases[0]), missing, bases[0])
        return ScalarAggregate(nd.asarray(x).view_scalars(dshape), **kwargs)
    return finalize


def as_float64(arr):
    return np.where(is_missing(arr), np.nan, arr.astype('f8'))


def finalize_mean(bases, **kwargs):
    sums, counts = bases
    with np.errstate(divide='ignore', invalid='ignore'):
        x = as_float64(sums)/counts
    return ScalarAggregate(nd.asarray(x).view_scalars('?float64'), **kwargs)


def finalize_var(bases, **kwargs):
    sums, counts, m2s = bases
    with np.errstate(divide='ignore', invalid='ignore'):
        x = as_float64(m2s)/counts
    return ScalarAggregate(nd.asarray(x).view_scalars('?float64'), **kwargs)


def finalize_std(bases, **kwargs):
    sums, counts, m2s = bases
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.sqrt(as_float64(m2s)/counts)
    return ScalarAggregate(nd.asarray(x).view_scalars('?float64'), **kwargs)
