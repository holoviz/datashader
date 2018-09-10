from __future__ import absolute_import, division, print_function

import numpy as np
from datashape import dshape, isnumeric, Record, Option
from datashape import coretypes as ct
from toolz import concat, unique
import xarray as xr

from .utils import Expr, ngjit


class Preprocess(Expr):
    """Base clase for preprocessing steps."""
    def __init__(self, column):
        self.column = column

    @property
    def inputs(self):
        return (self.column,)


class extract(Preprocess):
    """Extract a column from a dataframe as a numpy array of values."""
    def apply(self, df):
        return df[self.column].values


class category_codes(Preprocess):
    """Extract just the category codes from a categorical column."""
    def apply(self, df):
        return df[self.column].cat.codes.values


class Reduction(Expr):
    """Base class for per-bin reductions."""
    def __init__(self, column=None):
        self.column = column

    def validate(self, in_dshape):
        if not self.column in in_dshape.dict:
            raise ValueError("specified column not found")
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


class OptionalFieldReduction(Reduction):
    """Base class for things like ``count`` or ``any``"""
    def __init__(self, column=None):
        self.column = column

    @property
    def inputs(self):
        return (extract(self.column),) if self.column is not None else ()

    def validate(self, in_dshape):
        pass

    def _build_append(self, dshape):
        return self._append if self.column is None else self._append_non_na

    @staticmethod
    def _finalize(bases, **kwargs):
        return xr.DataArray(bases[0], **kwargs)


class count(OptionalFieldReduction):
    """Count elements in each bin.

    Parameters
    ----------
    column : str, optional
        If provided, only counts elements in ``column`` that are not ``NaN``.
        Otherwise, counts every element.
    """
    _dshape = dshape(ct.int32)

    @staticmethod
    @ngjit
    def _append(x, y, agg):
        agg[y, x] += 1

    @staticmethod
    @ngjit
    def _append_non_na(x, y, agg, field):
        if not np.isnan(field):
            agg[y, x] += 1

    @staticmethod
    def _create(shape):
        return np.zeros(shape, dtype='i4')

    @staticmethod
    def _combine(aggs):
        return aggs.sum(axis=0, dtype='i4')


class any(OptionalFieldReduction):
    """Whether any elements in ``column`` map to each bin.

    Parameters
    ----------
    column : str, optional
        If provided, only elements in ``column`` that are ``NaN`` are skipped.
    """
    _dshape = dshape(ct.bool_)

    @staticmethod
    @ngjit
    def _append(x, y, agg):
        agg[y, x] = True

    @staticmethod
    @ngjit
    def _append_non_na(x, y, agg, field):
        if not np.isnan(field):
            agg[y, x] = True

    @staticmethod
    def _create(shape):
        return np.zeros(shape, dtype='bool')

    @staticmethod
    def _combine(aggs):
        return aggs.sum(axis=0, dtype='bool')


class FloatingReduction(Reduction):
    """Base classes for reductions that always have floating-point dtype."""
    _dshape = dshape(Option(ct.float64))

    @staticmethod
    def _create(shape):
        return np.full(shape, np.nan, dtype='f8')

    @staticmethod
    def _finalize(bases, **kwargs):
        return xr.DataArray(bases[0], **kwargs)


class sum(FloatingReduction):
    """Sum of all elements in ``column``.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """
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
    """Sum of square differences from the mean of all elements in ``column``.

    Intermediate value for computing ``var`` and ``std``, not intended to be
    used on its own.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """
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
        with np.errstate(divide='ignore', invalid='ignore'):
            mu = np.nansum(sums, axis=0) / ns.sum(axis=0)
            return np.nansum(Ms + ns*(sums/ns - mu)**2, axis=0)


class min(FloatingReduction):
    """Minimum value of all elements in ``column``.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """
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
    """Maximum value of all elements in ``column``.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """
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
    """Count of all elements in ``column``, grouped by category.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be
        categorical. Resulting aggregate has a outer dimension axis along the
        categories present.
    """
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
    """Mean of all elements in ``column``.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """
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
    """Variance of all elements in ``column``.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """
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
    """Standard Deviation of all elements in ``column``.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """
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


class first(Reduction):
    """First value encountered in ``column``.

    Useful for categorical data where an actual value must always be returned, 
    not an average or other numerical calculation.
    
    Currently only supported for rasters, externally to this class.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. If the data type is floating point, 
        ``NaN`` values in the column are skipped.
    """
    _dshape = dshape(Option(ct.float64))

    @staticmethod 
    def _append(x, y, agg):
        raise NotImplementedError("first is currently implemented only for rasters")
    
    @staticmethod 
    def _create(shape):
        raise NotImplementedError("first is currently implemented only for rasters")

    @staticmethod
    def _combine(aggs):
        raise NotImplementedError("first is currently implemented only for rasters")

    @staticmethod
    def _finalize(bases, **kwargs):
        raise NotImplementedError("first is currently implemented only for rasters")



class last(Reduction):
    """Last value encountered in ``column``.

    Useful for categorical data where an actual value must always be returned, 
    not an average or other numerical calculation.
    
    Currently only supported for rasters, externally to this class.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. If the data type is floating point, 
        ``NaN`` values in the column are skipped.
    """
    _dshape = dshape(Option(ct.float64))

    @staticmethod 
    def _append(x, y, agg):
        raise NotImplementedError("last is currently implemented only for rasters")
    
    @staticmethod 
    def _create(shape):
        raise NotImplementedError("last is currently implemented only for rasters")

    @staticmethod
    def _combine(aggs):
        raise NotImplementedError("last is currently implemented only for rasters")

    @staticmethod
    def _finalize(bases, **kwargs):
        raise NotImplementedError("last is currently implemented only for rasters")



class mode(Reduction):
    """Mode (most common value) of all the values encountered in ``column``.

    Useful for categorical data where an actual value must always be returned, 
    not an average or other numerical calculation.
    
    Currently only supported for rasters, externally to this class.
    Implementing it for other glyph types would be difficult due to potentially
    unbounded data storage requirements to store indefinite point or line
    data per pixel.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. If the data type is floating point, 
        ``NaN`` values in the column are skipped.
    """
    _dshape = dshape(Option(ct.float64))

    @staticmethod 
    def _append(x, y, agg):
        raise NotImplementedError("mode is currently implemented only for rasters")
    
    @staticmethod 
    def _create(shape):
        raise NotImplementedError("mode is currently implemented only for rasters")

    @staticmethod
    def _combine(aggs):
        raise NotImplementedError("mode is currently implemented only for rasters")

    @staticmethod
    def _finalize(bases, **kwargs):
        raise NotImplementedError("mode is currently implemented only for rasters")



class summary(Expr):
    """A collection of named reductions.

    Computes all aggregates simultaneously, output is stored as a
    ``xarray.Dataset``.

    Examples
    --------
    A reduction for computing the mean of column "a", and the sum of column "b"
    for each bin, all in a single pass.

    >>> import datashader as ds
    >>> red = ds.summary(mean_a=ds.mean('a'), sum_b=ds.sum('b'))
    """
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



__all__ = list(set([_k for _k,_v in locals().items()
                    if isinstance(_v,type) and (issubclass(_v,Reduction) or _v is summary)
                    and _v not in [Reduction, OptionalFieldReduction,
                                   FloatingReduction, m2]]))
    
