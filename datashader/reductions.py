from __future__ import absolute_import, division, print_function

import numpy as np
from datashape import dshape, isnumeric, Record, Option
from datashape import coretypes as ct
from toolz import concat, unique
import xarray as xr

from .utils import ngjit


class Expr(object):
    def __init__(self, column):
        self.column = column

    def __hash__(self):
        return hash((type(self), self.inputs))

    def __eq__(self, other):
        return type(self) is type(other) and self.inputs == other.inputs

    def __ne__(self, other):
        return not self == other


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
        return self._dshape

    @property
    def inputs(self):
        return (extract(self.column),)

    @property
    def _bases(self):
        return (self,)

    @property
    def _temps(self):
        return ()

    def _build_create(self, dshape):
        return self._create

    def _build_append(self, dshape):
        return self._append

    def _build_combine(self, dshape):
        return self._combine

    def _build_finalize(self, dshape):
        return self._finalize


@ngjit
def append_count(x, y, agg):
    agg[y, x] += 1


@ngjit
def append_count_non_na(x, y, agg, field):
    if not np.isnan(field):
        agg[y, x] += 1


class count(Reduction):
    _dshape = dshape(ct.int32)

    def __init__(self, column=None):
        self.column = column

    @property
    def inputs(self):
        return (extract(self.column),) if self.column else ()

    def validate(self, in_dshape):
        pass

    @staticmethod
    def _create(shape):
        return np.zeros(shape, dtype='i4')

    def _build_append(self, dshape):
        return append_count if self.column is None else append_count_non_na

    @staticmethod
    def _combine(aggs):
        return aggs.sum(axis=0, dtype='i4')

    @staticmethod
    def _finalize(bases, **kwargs):
        return xr.DataArray(bases[0], **kwargs)


class FloatingReduction(Reduction):
    _dshape = dshape(Option(ct.float64))

    @staticmethod
    def _create(shape):
        return np.full(shape, np.nan, dtype='f8')

    @staticmethod
    def _finalize(bases, **kwargs):
        return xr.DataArray(bases[0], **kwargs)


class sum(FloatingReduction):
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not np.isnan(field):
            if np.isnan(agg[y, x]):
                agg[y, x] = field
            else:
                agg[y, x] += field

    @staticmethod
    def _combine(aggs):
        missing_vals = np.isnan(aggs)
        all_empty = np.bitwise_and.reduce(missing_vals, axis=0)
        set_to_zero = missing_vals & ~all_empty
        return np.where(set_to_zero, 0, aggs).sum(axis=0)


class m2(FloatingReduction):
    """Second moment"""
    @property
    def _temps(self):
        return (sum(self.column), count(self.column))

    @staticmethod
    @ngjit
    def _append(x, y, m2, field, sum, count):
        # sum & count are the results of sum[y, x], count[y, x] before being
        # updated by field
        if not np.isnan(field):
            if count == 0:
                m2[y, x] = 0
            else:
                u1 = np.float64(sum) / count
                u = np.float64(sum + field) / (count + 1)
                m2[y, x] += (field - u1) * (field - u)

    @staticmethod
    def _combine(Ms, sums, ns):
        mu = np.nansum(sums, axis=0) / ns.sum(axis=0)
        return np.nansum(Ms + ns*(sums/ns - mu)**2, axis=0)


class min(FloatingReduction):
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if np.isnan(agg[y, x]):
            agg[y, x] = field
        elif agg[y, x] > field:
            agg[y, x] = field

    @staticmethod
    def _combine(aggs):
        return np.nanmin(aggs, axis=0)


class max(FloatingReduction):
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if np.isnan(agg[y, x]):
            agg[y, x] = field
        elif agg[y, x] < field:
            agg[y, x] = field

    @staticmethod
    def _combine(aggs):
        return np.nanmax(aggs, axis=0)


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

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        agg[y, x, field] += 1

    @staticmethod
    def _combine(aggs):
        return aggs.sum(axis=0, dtype='i4')

    def _build_finalize(self, dshape):
        cats = list(dshape[self.column].categories)

        def finalize(bases, **kwargs):
            dims = kwargs['dims'] + [self.column]
            coords = kwargs['coords'] + [cats]
            return xr.DataArray(bases[0], dims=dims, coords=coords)
        return finalize


class mean(Reduction):
    _dshape = dshape(Option(ct.float64))

    @property
    def _bases(self):
        return (sum(self.column), count(self.column))

    @staticmethod
    def _finalize(bases, **kwargs):
        sums, counts = bases
        with np.errstate(divide='ignore', invalid='ignore'):
            x = sums/counts
        return xr.DataArray(x, **kwargs)


class var(Reduction):
    _dshape = dshape(Option(ct.float64))

    @property
    def _bases(self):
        return (sum(self.column), count(self.column), m2(self.column))

    @staticmethod
    def _finalize(bases, **kwargs):
        sums, counts, m2s = bases
        with np.errstate(divide='ignore', invalid='ignore'):
            x = m2s/counts
        return xr.DataArray(x, **kwargs)


class std(Reduction):
    _dshape = dshape(Option(ct.float64))

    @property
    def _bases(self):
        return (sum(self.column), count(self.column), m2(self.column))

    @staticmethod
    def _finalize(bases, **kwargs):
        sums, counts, m2s = bases
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.sqrt(m2s/counts)
        return xr.DataArray(x, **kwargs)


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
