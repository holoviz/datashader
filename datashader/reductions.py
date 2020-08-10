from __future__ import absolute_import, division, print_function

import numpy as np
from datashape import dshape, isnumeric, Record, Option
from datashape import coretypes as ct
from toolz import concat, unique
import xarray as xr

from datashader.glyphs.glyph import isnull
from numba import cuda as nb_cuda
from datashader.transfer_functions._cuda_utils import (cuda_atomic_nanmin,
                                                       cuda_atomic_nanmax)

try:
    import cudf
except Exception:
    cudf = None

from .utils import Expr, ngjit, nansum_missing


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
        if cudf and isinstance(df, cudf.DataFrame):
            import cupy
            if df[self.column].dtype.kind == 'f':
                nullval = np.nan
            else:
                nullval = 0
            return cupy.array(df[self.column].to_gpu_array(fillna=nullval))
        elif isinstance(df, xr.Dataset):
            # DataArray could be backed by numpy or cupy array
            return df[self.column].data
        else:
            return df[self.column].values


class category_codes(Preprocess):
    """Extract just the category codes from a categorical column."""
    def apply(self, df):
        if cudf and isinstance(df, cudf.DataFrame):
            return df[self.column].cat.codes.to_gpu_array()
        else:
            return df[self.column].cat.codes.values

class category_values(Preprocess):
    """Extract multiple columns from a dataframe as a numpy array of values."""
    def __init__(self, columns):
        self.columns = list(columns)

    @property
    def inputs(self):
        return self.columns

    def apply(self, df):
        if cudf and isinstance(df, cudf.DataFrame):
            import cupy
            if df[self.columns[1]].dtype.kind == 'f':
                nullval = np.nan
            else:
                nullval = 0
            a = cupy.asarray(df[self.columns[0]].cat.codes.to_gpu_array())
            b = cupy.asarray(df[self.columns[1]].to_gpu_array(fillna=nullval))
            return cupy.stack((a, b), axis=-1)
        else:
            a = df[self.columns[0]].cat.codes.values
            b = df[self.columns[1]].values
            return np.stack((a, b), axis=-1)

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

    def _build_bases(self, cuda=False):
        return (self,)

    def _build_temps(self, cuda=False):
        return ()

    def _build_create(self, dshape):
        return self._create

    def _build_append(self, dshape, schema, cuda=False):
        if cuda:
            if self.column is None:
                return self._append_no_field_cuda
            else:
                return self._append_cuda
        else:
            if self.column is None:
                return self._append_no_field
            else:
                return self._append

    def _build_combine(self, dshape):
        return self._combine

    def _build_finalize(self, dshape):
        return self._finalize


class OptionalFieldReduction(Reduction):
    """Base class for things like ``count`` or ``any`` for which the field is optional"""
    def __init__(self, column=None):
        self.column = column

    @property
    def inputs(self):
        return (extract(self.column),) if self.column is not None else ()

    def validate(self, in_dshape):
        pass

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        return xr.DataArray(bases[0], **kwargs)

class by(Reduction):
    """Apply the provided reduction separately per categorical ``column`` value.
    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be
        categorical. Resulting aggregate has an outer dimension axis along the
        categories present.
    reduction : Reduction
        Per-category reduction function.
    """
    def __init__(self, cat_column, reduction):
        self.columns = (cat_column, getattr(reduction, 'column', None))
        self.reduction = reduction
        self.column = cat_column # for backwards compatibility with count_cat

    def __hash__(self):
        return hash((type(self), self._hashable_inputs(), self.reduction))

    def _build_temps(self, cuda=False):
        return tuple(by(self.cat_column, tmp) for tmp in self.reduction._build_temps(cuda))

    @property
    def cat_column(self):
        return self.columns[0]

    @property
    def val_column(self):
        return self.columns[1]

    def validate(self, in_dshape):
        if not self.cat_column in in_dshape.dict:
            raise ValueError("specified column not found")
        if not isinstance(in_dshape.measure[self.cat_column], ct.Categorical):
            raise ValueError("input must be categorical")

        self.reduction.validate(in_dshape)

    def out_dshape(self, input_dshape):
        cats = input_dshape.measure[self.cat_column].categories
        red_shape = self.reduction.out_dshape(input_dshape)
        return dshape(Record([(c, red_shape) for c in cats]))

    @property
    def inputs(self):
        if self.val_column is not None:
            return (category_values(self.columns),)
        else:
            return (category_codes(self.columns[0]),)

    def _build_create(self, out_dshape):
        n_cats = len(out_dshape.measure.fields)
        return lambda shape, array_module: self.reduction._build_create(
            out_dshape)(shape + (n_cats,), array_module)

    def _build_bases(self, cuda=False):
        bases = self.reduction._build_bases(cuda)
        if len(bases) == 1 and bases[0] is self:
            return bases
        return tuple(by(self.cat_column, base) for base in bases)

    def _build_append(self, dshape, schema, cuda=False):
        return self.reduction._build_append(dshape, schema, cuda)

    def _build_combine(self, dshape):
        return self.reduction._combine

    def _build_finalize(self, dshape):
        cats = list(dshape[self.cat_column].categories)

        def finalize(bases, cuda=False, **kwargs):
            kwargs['dims'] += [self.cat_column]
            kwargs['coords'][self.cat_column] = cats
            return self.reduction._finalize(bases, cuda=cuda, **kwargs)

        return finalize

class count(OptionalFieldReduction):
    """Count elements in each bin, returning the result as a uint32.

    Parameters
    ----------
    column : str, optional
        If provided, only counts elements in ``column`` that are not ``NaN``.
        Otherwise, counts every element.
    """
    _dshape = dshape(ct.uint32)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append_no_field(x, y, agg):
        agg[y, x] += 1


    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            agg[y, x] += 1

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_no_field_cuda(x, y, agg):
        nb_cuda.atomic.add(agg, (y, x), 1)

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if not isnull(field):
            nb_cuda.atomic.add(agg, (y, x), 1)

    @staticmethod
    def _create(shape, array_module):
        return array_module.zeros(shape, dtype='u4')

    @staticmethod
    def _combine(aggs):
        return aggs.sum(axis=0, dtype='u4')


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
    def _append_no_field(x, y, agg):
        agg[y, x] = True
    _append_no_field_cuda = _append_no_field

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            agg[y, x] = True
    _append_cuda =_append

    @staticmethod
    def _create(shape, array_module):
        return array_module.zeros(shape, dtype='bool')

    @staticmethod
    def _combine(aggs):
        return aggs.sum(axis=0, dtype='bool')


class _upsample(Reduction):
    """"Special internal class used for upsampling"""
    _dshape = dshape(Option(ct.float64))

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        return xr.DataArray(bases[0], **kwargs)

    @property
    def inputs(self):
        return (extract(self.column),)

    @staticmethod
    def _create(shape, array_module):
        # Use uninitialized memory, the upsample function must explicitly set unused
        # values to nan
        return array_module.empty(shape, dtype='f8')

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        # not called, the upsample function must set agg directly
        pass

    @staticmethod
    @ngjit
    def _append_cuda(x, y, agg, field):
        # not called, the upsample function must set agg directly
        pass

    @staticmethod
    def _combine(aggs):
        return np.nanmax(aggs, axis=0)


class FloatingReduction(Reduction):
    """Base classes for reductions that always have floating-point dtype."""
    _dshape = dshape(Option(ct.float64))

    @staticmethod
    def _create(shape, array_module):
        return array_module.full(shape, np.nan, dtype='f8')

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        return xr.DataArray(bases[0], **kwargs)


class _sum_zero(FloatingReduction):
    """Sum of all elements in ``column``.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """

    @staticmethod
    def _create(shape, array_module):
        return array_module.zeros(shape, dtype='f8')

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            agg[y, x] += field

    @staticmethod
    @ngjit
    def _append_cuda(x, y, agg, field):
        if not isnull(field):
            nb_cuda.atomic.add(agg, (y, x), field)

    @staticmethod
    def _combine(aggs):
        return aggs.sum(axis=0, dtype='f8')

class sum(FloatingReduction):
    """Sum of all elements in ``column``.

    Elements of resulting aggregate are nan if they are not updated.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """
    _dshape = dshape(Option(ct.float64))

    # Cuda implementation
    def _build_bases(self, cuda=False):
        if cuda:
            return (_sum_zero(self.column), any(self.column))
        else:
            return (self,)

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        if cuda:
            sums, anys = bases
            x = np.where(anys, sums, np.nan)
            return xr.DataArray(x, **kwargs)
        else:
            return xr.DataArray(bases[0], **kwargs)

    # Single pass CPU implementation
    # These methods will only be called if _build_bases returned (self,)
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            if isnull(agg[y, x]):
                agg[y, x] = field
            else:
                agg[y, x] += field

    @staticmethod
    def _combine(aggs):
        return nansum_missing(aggs, axis=0)


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

    @staticmethod
    def _create(shape, array_module):
        return array_module.full(shape, 0.0, dtype='f8')

    def _build_temps(self, cuda=False):
        return (_sum_zero(self.column), count(self.column))

    def _build_append(self, dshape, schema, cuda=False):
        if cuda:
            raise ValueError("""\
The 'std' and 'var' reduction operations are not yet supported on the GPU""")
        return super(m2, self)._build_append(dshape, schema, cuda)

    @staticmethod
    @ngjit
    def _append(x, y, m2, field, sum, count):
        # sum & count are the results of sum[y, x], count[y, x] before being
        # updated by field
        if not isnull(field):
            if count > 0:
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
        if isnull(agg[y, x]):
            agg[y, x] = field
        elif agg[y, x] > field:
            agg[y, x] = field

    @staticmethod
    @ngjit
    def _append_cuda(x, y, agg, field):
        cuda_atomic_nanmin(agg, (y, x), field)

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
        if isnull(agg[y, x]):
            agg[y, x] = field
        elif agg[y, x] < field:
            agg[y, x] = field

    @staticmethod
    @ngjit
    def _append_cuda(x, y, agg, field):
        cuda_atomic_nanmax(agg, (y, x), field)

    @staticmethod
    def _combine(aggs):
        return np.nanmax(aggs, axis=0)


class count_cat(by):
    """Count of all elements in ``column``, grouped by category.
    Alias for `by(...,count())`, for backwards compatibility.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be
        categorical. Resulting aggregate has a outer dimension axis along the
        categories present.
    """
    def __init__(self, column):
        super(count_cat, self).__init__(column, count())


class mean(Reduction):
    """Mean of all elements in ``column``.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """
    _dshape = dshape(Option(ct.float64))

    def _build_bases(self, cuda=False):
        return (_sum_zero(self.column), count(self.column))

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        sums, counts = bases
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.where(counts > 0, sums/counts, np.nan)
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

    def _build_bases(self, cuda=False):
        return (_sum_zero(self.column), count(self.column), m2(self.column))

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        sums, counts, m2s = bases
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.where(counts > 0, m2s / counts, np.nan)
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

    def _build_bases(self, cuda=False):
        return (_sum_zero(self.column), count(self.column), m2(self.column))

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        sums, counts, m2s = bases
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.where(counts > 0, np.sqrt(m2s / counts), np.nan)
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
    def _create(shape, array_module):
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
    def _create(shape, array_module):
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
    def _create(shape, array_module):
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
