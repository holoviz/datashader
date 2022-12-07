from __future__ import annotations
from packaging.version import Version
import numpy as np
from datashape import dshape, isnumeric, Record, Option
from datashape import coretypes as ct
from toolz import concat, unique
import xarray as xr

from datashader.enums import AntialiasCombination
from datashader.utils import isnull
from numba import cuda as nb_cuda

try:
    from datashader.transfer_functions._cuda_utils import (cuda_atomic_nanmin,
                                                           cuda_atomic_nanmax)
except ImportError:
    cuda_atomic_nanmin, cuda_atomic_nanmmax = None, None

try:
    import cudf
    import cupy as cp
except Exception:
    cudf = cp = None

from .utils import Expr, ngjit, nansum_missing, nanmax_in_place, nansum_in_place


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
            if df[self.column].dtype.kind == 'f':
                nullval = np.nan
            else:
                nullval = 0
            if Version(cudf.__version__) >= Version("22.02"):
                return df[self.column].to_cupy(na_value=nullval)
            return cp.array(df[self.column].to_gpu_array(fillna=nullval))
        elif isinstance(df, xr.Dataset):
            # DataArray could be backed by numpy or cupy array
            return df[self.column].data
        else:
            return df[self.column].values

class CategoryPreprocess(Preprocess):
    """Base class for categorizing preprocessors."""
    @property
    def cat_column(self):
        """Returns name of categorized column"""
        return self.column

    def categories(self, input_dshape):
        """Returns list of categories corresponding to input shape"""
        raise NotImplementedError("categories not implemented")

    def validate(self, in_dshape):
        """Validates input shape"""
        raise NotImplementedError("validate not implemented")

    def apply(self, df):
        """Applies preprocessor to DataFrame and returns array"""
        raise NotImplementedError("apply not implemented")


class category_codes(CategoryPreprocess):
    """
    Extract just the category codes from a categorical column.

    To create a new type of categorizer, derive a subclass from this
    class or one of its subclasses, implementing ``__init__``,
    ``_hashable_inputs``, ``categories``, ``validate``, and ``apply``.

    See the implementation of ``category_modulo`` in ``reductions.py``
    for an example.
    """
    def categories(self, input_dshape):
        return input_dshape.measure[self.column].categories

    def validate(self, in_dshape):
        if not self.column in in_dshape.dict:
            raise ValueError("specified column not found")
        if not isinstance(in_dshape.measure[self.column], ct.Categorical):
            raise ValueError("input must be categorical")

    def apply(self, df):
        if cudf and isinstance(df, cudf.DataFrame):
            if Version(cudf.__version__) >= Version("22.02"):
                return df[self.column].cat.codes.to_cupy()
            return df[self.column].cat.codes.to_gpu_array()
        else:
            return df[self.column].cat.codes.values

class category_modulo(category_codes):
    """
    A variation on category_codes that assigns categories using an integer column, modulo a base.
    Category is computed as (column_value - offset)%modulo.
    """

    # couldn't find anything in the datashape docs about how to check if a CType is an integer, so just define a big set
    IntegerTypes = {ct.bool_, ct.uint8, ct.uint16, ct.uint32, ct.uint64, ct.int8, ct.int16, ct.int32, ct.int64}

    def __init__(self, column, modulo, offset=0):
        super().__init__(column)
        self.offset = offset
        self.modulo = modulo

    def _hashable_inputs(self):
        return super()._hashable_inputs() + (self.offset, self.modulo)

    def categories(self, in_dshape):
        return list(range(self.modulo))

    def validate(self, in_dshape):
        if not self.column in in_dshape.dict:
            raise ValueError("specified column not found")
        if in_dshape.measure[self.column] not in self.IntegerTypes:
            raise ValueError("input must be an integer column")

    def apply(self, df):
        result = (df[self.column] - self.offset) % self.modulo
        if cudf and isinstance(df, cudf.Series):
            if Version(cudf.__version__) >= Version("22.02"):
                return result.to_cupy()
            return result.to_gpu_array()
        else:
            return result.values

class category_binning(category_modulo):
    """
    A variation on category_codes that assigns categories by binning a continuous-valued column.
    The number of categories returned is always nbins+1.
    The last category (nbin) is for NaNs in the data column, as well as for values under/over the binned
    interval (when include_under or include_over is False).

    Parameters
    ----------
    column:   column to use
    lower:    lower bound of first bin
    upper:    upper bound of last bin
    nbins:     number of bins
    include_under: if True, values below bin 0 are assigned to category 0
    include_over:  if True, values above the last bin (nbins-1) are assigned to category nbin-1
    """

    def __init__(self, column, lower, upper, nbins, include_under=True, include_over=True):
        super().__init__(column, nbins + 1)  # +1 category for NaNs and clipped values
        self.bin0 = lower
        self.binsize = (upper - lower) / float(nbins)
        self.nbins = nbins
        self.bin_under = 0 if include_under else nbins
        self.bin_over  = nbins-1 if include_over else nbins

    def _hashable_inputs(self):
        return super()._hashable_inputs() + (self.bin0, self.binsize, self.bin_under, self.bin_over)

    def validate(self, in_dshape):
        if not self.column in in_dshape.dict:
            raise ValueError("specified column not found")

    def apply(self, df):
        if cudf and isinstance(df, cudf.DataFrame):
            if Version(cudf.__version__) >= Version("22.02"):
                values = df[self.column].to_cupy(na_value=cp.nan)
            else:
                values = cp.array(df[self.column].to_gpu_array(fillna=True))
            nan_values = cp.isnan(values)
        else:
            values = df[self.column].to_numpy()
            nan_values = np.isnan(values)

        index = ((values - self.bin0) / self.binsize).astype(int)
        index[index < 0] = self.bin_under
        index[index >= self.nbins] = self.bin_over
        index[nan_values] = self.nbins
        return index


class category_values(CategoryPreprocess):
    """Extract a category and a value column from a dataframe as (2,N) numpy array of values."""
    def __init__(self, categorizer, value_column):
        super().__init__(value_column)
        self.categorizer = categorizer

    @property
    def inputs(self):
        return (self.categorizer.column, self.column)

    @property
    def cat_column(self):
        """Returns name of categorized column"""
        return self.categorizer.column

    def categories(self, input_dshape):
        return self.categorizer.categories

    def validate(self, in_dshape):
        return self.categorizer.validate(in_dshape)

    def apply(self, df):
        a = self.categorizer.apply(df)
        if cudf and isinstance(df, cudf.DataFrame):
            import cupy
            if df[self.column].dtype.kind == 'f':
                nullval = np.nan
            else:
                nullval = 0
            a = cupy.asarray(a)
            if Version(cudf.__version__) >= Version("22.02"):
                b = df[self.column].to_cupy(na_value=nullval)
            else:
                b = cupy.asarray(df[self.column].fillna(nullval))
            return cupy.stack((a, b), axis=-1)
        else:
            b = df[self.column].values
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

    @property
    def inputs(self):
        return (extract(self.column),)

    def _antialias_requires_2_stages(self):
        # Return True if this Reduction must be processed with 2 stages,
        # False if it doesn't matter.
        # Overridden in derived classes as appropriate.
        return False

    def _antialias_stage_2(self, self_intersect, array_module):
        # Only called if using antialiased lines. Overridden in derived classes.
        # Returns a tuple containing an item for each constituent reduction.
        # Each item is (AntialiasCombination, zero_value)).
        raise NotImplementedError(f"{type(self)}._antialias_stage_2 is not defined")

    def _build_bases(self, cuda=False):
        return (self,)

    def _build_temps(self, cuda=False):
        return ()

    def _build_create(self, required_dshape):
        fields = getattr(required_dshape.measure, "fields", None)
        if fields is not None and len(required_dshape.measure.fields) > 1:
            # If more than one field then they all have the same dtype so can just take the first.
            first_field = required_dshape.measure.fields[0]
            required_dshape = dshape(first_field[1])

        if isinstance(required_dshape, Option):
            required_dshape = dshape(required_dshape.ty)

        if required_dshape == dshape(ct.bool_):
            return self._create_bool
        elif required_dshape == dshape(ct.float32):
            return self._create_float32_nan
        elif required_dshape == dshape(ct.float64):
            return self._create_float64_nan
        elif required_dshape == dshape(ct.uint32):
            return self._create_uint32
        else:
            raise NotImplementedError(f"Unexpected dshape {dshape}")

    def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
        if cuda:
            if antialias and self.column is None:
                return self._append_no_field_antialias_cuda
            elif antialias:
                return self._append_antialias_cuda
            elif self.column is None:
                return self._append_no_field_cuda
            else:
                return self._append_cuda
        else:
            if antialias and self.column is None:
                return self._append_no_field_antialias
            elif antialias:
                return self._append_antialias
            elif self.column is None:
                return self._append_no_field
            else:
                return self._append

    def _build_combine(self, dshape, antialias):
        return self._combine

    def _build_finalize(self, dshape):
        return self._finalize

    @staticmethod
    def _create_bool(shape, array_module):
        return array_module.zeros(shape, dtype='bool')

    @staticmethod
    def _create_float32_nan(shape, array_module):
        return array_module.full(shape, array_module.nan, dtype='f4')

    @staticmethod
    def _create_float64_nan(shape, array_module):
        return array_module.full(shape, array_module.nan, dtype='f8')

    @staticmethod
    def _create_float64_empty(shape, array_module):
        return array_module.empty(shape, dtype='f8')

    @staticmethod
    def _create_float64_zero(shape, array_module):
        return array_module.zeros(shape, dtype='f8')

    @staticmethod
    def _create_uint32(shape, array_module):
        return array_module.zeros(shape, dtype='u4')


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


class SelfIntersectingOptionalFieldReduction(OptionalFieldReduction):
    """
    Base class for optional field reductions for which self-intersecting
    geometry may or may not be desireable.
    Ignored if not using antialiasing.
    """
    def __init__(self, column=None, self_intersect=True):
        super().__init__(column)
        self.self_intersect = self_intersect

    def _antialias_requires_2_stages(self):
        return not self.self_intersect

    def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
        if antialias and not self_intersect:
            # append functions specific to antialiased lines without self_intersect
            if cuda:
                if self.column is None:
                    return self._append_no_field_antialias_cuda_not_self_intersect
                else:
                    return self._append_antialias_cuda_not_self_intersect
            else:
                if self.column is None:
                    return self._append_no_field_antialias_not_self_intersect
                else:
                    return self._append_antialias_not_self_intersect

        # Fall back to base class implementation
        return super()._build_append(dshape, schema, cuda, antialias, self_intersect)

    def _hashable_inputs(self):
        # Reductions with different self_intersect attributes much have different hashes otherwise
        # toolz.memoize will treat them as the same to give incorrect results.
        return super()._hashable_inputs() + (self.self_intersect,)


class count(SelfIntersectingOptionalFieldReduction):
    """Count elements in each bin, returning the result as a uint32, or a
    float32 if using antialiasing.

    Parameters
    ----------
    column : str, optional
        If provided, only counts elements in ``column`` that are not ``NaN``.
        Otherwise, counts every element.
    """
    def out_dshape(self, in_dshape, antialias):
        return dshape(ct.float32) if antialias else dshape(ct.uint32)

    def _antialias_stage_2(self, self_intersect, array_module):
        if self_intersect:
            return ((AntialiasCombination.SUM_1AGG, array_module.nan),)
        else:
            return ((AntialiasCombination.SUM_2AGG, array_module.nan),)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            agg[y, x] += 1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        if not isnull(field):
            if isnull(agg[y, x]):
                agg[y, x] = aa_factor
            else:
                agg[y, x] += aa_factor

    @staticmethod
    @ngjit
    def _append_antialias_not_self_intersect(x, y, agg, field, aa_factor):
        if not isnull(field):
            if isnull(agg[y, x]) or aa_factor > agg[y, x]:
                agg[y, x] = aa_factor

    @staticmethod
    @ngjit
    def _append_no_field(x, y, agg):
        agg[y, x] += 1

    @staticmethod
    @ngjit
    def _append_no_field_antialias(x, y, agg, aa_factor):
        if isnull(agg[y, x]):
            agg[y, x] = aa_factor
        else:
            agg[y, x] += aa_factor

    @staticmethod
    @ngjit
    def _append_no_field_antialias_not_self_intersect(x, y, agg, aa_factor):
        if isnull(agg[y, x]) or aa_factor > agg[y, x]:
            agg[y, x] = aa_factor

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_antialias_cuda(x, y, agg, field, aa_factor):
        cuda_atomic_nanmax(agg, (y, x), field*aa_factor)

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_no_field_antialias_cuda_not_self_intersect(x, y, agg, aa_factor):
        cuda_atomic_nanmax(agg, (y, x), aa_factor)

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if not isnull(field):
            nb_cuda.atomic.add(agg, (y, x), 1)

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_no_field_antialias_cuda(x, y, agg, aa_factor):
        cuda_atomic_nanmax(agg, (y, x), aa_factor)

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_no_field_cuda(x, y, agg):
        nb_cuda.atomic.add(agg, (y, x), 1)

    def _build_combine(self, dshape, antialias):
        if antialias:
            return self._combine_antialias
        else:
            return self._combine

    @staticmethod
    def _combine(aggs):
        return aggs.sum(axis=0, dtype='u4')

    @staticmethod
    def _combine_antialias(aggs):
        ret = aggs[0]
        for i in range(1, len(aggs)):
            nansum_in_place(ret, aggs[i])
        return ret


class by(Reduction):
    """Apply the provided reduction separately per category.
    Parameters
    ----------
    cats: str or CategoryPreprocess instance
        Name of column to aggregate over, or a categorizer object that returns categories.
        Resulting aggregate has an outer dimension axis along the categories present.
    reduction : Reduction
        Per-category reduction function.
    """
    def __init__(self, cat_column, reduction=count()):
        # set basic categorizer
        if isinstance(cat_column, CategoryPreprocess):
            self.categorizer = cat_column
        elif isinstance(cat_column, str):
            self.categorizer = category_codes(cat_column)
        else:
            raise TypeError("first argument must be a column name or a CategoryPreprocess instance")
        self.column = self.categorizer.column # for backwards compatibility with count_cat
        self.columns = (self.categorizer.column, getattr(reduction, 'column', None))
        self.reduction = reduction
        # if a value column is supplied, set category_values preprocessor
        if self.val_column is not None:
            self.preprocess = category_values(self.categorizer, self.val_column)
        else:
            self.preprocess = self.categorizer

    def __hash__(self):
        return hash((type(self), self._hashable_inputs(), self.categorizer._hashable_inputs(), self.reduction))

    def _build_temps(self, cuda=False):
        return tuple(by(self.categorizer, tmp) for tmp in self.reduction._build_temps(cuda))

    @property
    def cat_column(self):
        return self.columns[0]

    @property
    def val_column(self):
        return self.columns[1]

    def validate(self, in_dshape):
        self.preprocess.validate(in_dshape)
        self.reduction.validate(in_dshape)

    def out_dshape(self, input_dshape, antialias):
        cats = self.categorizer.categories(input_dshape)
        red_shape = self.reduction.out_dshape(input_dshape, antialias)
        return dshape(Record([(c, red_shape) for c in cats]))

    @property
    def inputs(self):
        return (self.preprocess, )

    def _antialias_requires_2_stages(self):
        return self.reduction._antialias_requires_2_stages()

    def _antialias_stage_2(self, self_intersect, array_module):
        return self.reduction._antialias_stage_2(self_intersect, array_module)

    def _build_create(self, required_dshape):
        n_cats = len(required_dshape.measure.fields)
        return lambda shape, array_module: self.reduction._build_create(
            required_dshape)(shape + (n_cats,), array_module)

    def _build_bases(self, cuda=False):
        bases = self.reduction._build_bases(cuda)
        if len(bases) == 1 and bases[0] is self:
            return bases
        return tuple(by(self.categorizer, base) for base in bases)

    def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
        return self.reduction._build_append(dshape, schema, cuda, antialias, self_intersect)

    def _build_combine(self, dshape, antialias):
        return self.reduction._build_combine(dshape, antialias)

    def _build_finalize(self, dshape):
        cats = list(self.categorizer.categories(dshape))

        def finalize(bases, cuda=False, **kwargs):
            kwargs['dims'] += [self.cat_column]
            kwargs['coords'][self.cat_column] = cats
            return self.reduction._finalize(bases, cuda=cuda, **kwargs)

        return finalize

class any(OptionalFieldReduction):
    """Whether any elements in ``column`` map to each bin.

    Parameters
    ----------
    column : str, optional
        If provided, any elements in ``column`` that are ``NaN`` are skipped.
    """
    def out_dshape(self, in_dshape, antialias):
        return dshape(ct.float32) if antialias else dshape(ct.bool_)

    def _antialias_stage_2(self, self_intersect, array_module):
        return ((AntialiasCombination.MAX, array_module.nan),)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            agg[y, x] = True

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        if not isnull(field):
            if isnull(agg[y, x]) or aa_factor > agg[y, x]:
                agg[y, x] = aa_factor

    @staticmethod
    @ngjit
    def _append_no_field(x, y, agg):
        agg[y, x] = True

    @staticmethod
    @ngjit
    def _append_no_field_antialias(x, y, agg, aa_factor):
        if isnull(agg[y, x]) or aa_factor > agg[y, x]:
            agg[y, x] = aa_factor

    # GPU append functions
    _append_cuda =_append
    _append_no_field_cuda = _append_no_field

    def _build_combine(self, dshape, antialias):
        if antialias:
            return self._combine_antialias
        else:
            return self._combine

    @staticmethod
    def _combine(aggs):
        return aggs.sum(axis=0, dtype='bool')

    @staticmethod
    def _combine_antialias(aggs):
        ret = aggs[0]
        for i in range(1, len(aggs)):
            nanmax_in_place(ret, aggs[i])
        return ret


class _upsample(Reduction):
    """"Special internal class used for upsampling"""
    def out_dshape(self, in_dshape, antialias):
        return dshape(Option(ct.float64))

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        return xr.DataArray(bases[0], **kwargs)

    @property
    def inputs(self):
        return (extract(self.column),)

    def _build_create(self, required_dshape):
        # Use uninitialized memory, the upsample function must explicitly set unused
        # values to nan
        return self._create_float64_empty

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        # not called, the upsample function must set agg directly
        pass

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        # not called, the upsample function must set agg directly
        pass

    @staticmethod
    def _combine(aggs):
        return np.nanmax(aggs, axis=0)


class FloatingReduction(Reduction):
    """Base classes for reductions that always have floating-point dtype."""
    def out_dshape(self, in_dshape, antialias):
        return dshape(Option(ct.float64))

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        return xr.DataArray(bases[0], **kwargs)


class _sum_zero(FloatingReduction):
    """Sum of all elements in ``column``.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
    """
    def _antialias_stage_2(self, self_intersect, array_module):
        if self_intersect:
            return ((AntialiasCombination.SUM_1AGG, 0),)
        else:
            return ((AntialiasCombination.SUM_2AGG, 0),)

    def _build_create(self, required_dshape):
        return self._create_float64_zero

    # CPU append functions.
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            # agg[y, x] cannot be null as initialised to zero.
            agg[y, x] += field

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value):
            # agg[y, x] cannot be null as initialised to zero.
            agg[y, x] += value

    @staticmethod
    @ngjit
    def _append_antialias_not_self_intersect(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value) and value > agg[y, x]:
            # agg[y, x] cannot be null as initialised to zero.
            agg[y, x] = value

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if not isnull(field):
            nb_cuda.atomic.add(agg, (y, x), field)

    @staticmethod
    def _combine(aggs):
        return aggs.sum(axis=0, dtype='f8')


class SelfIntersectingFloatingReduction(FloatingReduction):
    """
    Base class fo floating reductions for which self-intersecting geometry
    may or may not be desireable.
    Ignored if not using antialiasing.
    """
    def __init__(self, column=None, self_intersect=True):
        super().__init__(column)
        self.self_intersect = self_intersect

    def _antialias_requires_2_stages(self):
        return not self.self_intersect

    def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
        if antialias and not self_intersect:
            if cuda:
                raise NotImplementedError("SelfIntersectingOptionalFieldReduction")
            else:
                if self.column is None:
                    return self._append_no_field_antialias_not_self_intersect
                else:
                    return self._append_antialias_not_self_intersect

        return super()._build_append(dshape, schema, cuda, antialias, self_intersect)

    def _hashable_inputs(self):
        # Reductions with different self_intersect attributes much have different hashes otherwise
        # toolz.memoize will treat them as the same to give incorrect results.
        return super()._hashable_inputs() + (self.self_intersect,)


class sum(SelfIntersectingFloatingReduction):
    """Sum of all elements in ``column``.

    Elements of resulting aggregate are nan if they are not updated.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """
    def _antialias_stage_2(self, self_intersect, array_module):
        if self_intersect:
            return ((AntialiasCombination.SUM_1AGG, array_module.nan),)
        else:
            return ((AntialiasCombination.SUM_2AGG, array_module.nan),)

    def _build_bases(self, cuda=False):
        if cuda:
            return (_sum_zero(self.column), any(self.column))
        else:
            return (self,)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            if isnull(agg[y, x]):
                agg[y, x] = field
            else:
                agg[y, x] += field

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value):
            if isnull(agg[y, x]):
                agg[y, x] = value
            else:
                agg[y, x] += value

    @staticmethod
    @ngjit
    def _append_antialias_not_self_intersect(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value):
            if isnull(agg[y, x]) or value > agg[y, x]:
                agg[y, x] = value

    @staticmethod
    def _combine(aggs):
        return nansum_missing(aggs, axis=0)

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        if cuda:
            sums, anys = bases
            x = np.where(anys, sums, np.nan)
            return xr.DataArray(x, **kwargs)
        else:
            return xr.DataArray(bases[0], **kwargs)


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
    def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
        if cuda:
            raise ValueError("""\
The 'std' and 'var' reduction operations are not yet supported on the GPU""")
        return super(m2, self)._build_append(dshape, schema, cuda, antialias, self_intersect)

    def _build_create(self, required_dshape):
        return self._create_float64_zero

    def _build_temps(self, cuda=False):
        return (_sum_zero(self.column), count(self.column))

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
    def _antialias_requires_2_stages(self):
        return True

    def _antialias_stage_2(self, self_intersect, array_module):
        return ((AntialiasCombination.MIN, array_module.nan),)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if isnull(agg[y, x]):
            agg[y, x] = field
        elif agg[y, x] > field:
            agg[y, x] = field

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if isnull(agg[y, x]) or value > agg[y, x]:
            agg[y, x] = value

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
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
    def _antialias_stage_2(self, self_intersect, array_module):
        return ((AntialiasCombination.MAX, array_module.nan),)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if isnull(agg[y, x]):
            agg[y, x] = field
        elif agg[y, x] < field:
            agg[y, x] = field

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if isnull(agg[y, x]) or value > agg[y, x]:
            agg[y, x] = value

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_antialias_cuda(x, y, agg, field, aa_factor):
        cuda_atomic_nanmax(agg, (y, x), field*aa_factor)

    @staticmethod
    @nb_cuda.jit(device=True)
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
    def out_dshape(self, in_dshape, antialias):
        return dshape(Option(ct.float64))

    def _antialias_requires_2_stages(self):
        return True

    def _antialias_stage_2(self, self_intersect, array_module):
        return ((AntialiasCombination.FIRST, array_module.nan),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field) and isnull(agg[y, x]):
            agg[y, x] = field

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if isnull(agg[y, x]) or value > agg[y, x]:
            agg[y, x] = value

    @staticmethod
    def _combine(aggs):
        raise NotImplementedError("first is not implemented for dask DataFrames")

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        return xr.DataArray(bases[0], **kwargs)



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
    def out_dshape(self, in_dshape, antialias):
        return dshape(Option(ct.float64))

    def _antialias_requires_2_stages(self):
        return True

    def _antialias_stage_2(self, self_intersect, array_module):
        return ((AntialiasCombination.LAST, array_module.nan),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            agg[y, x] = field

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if isnull(agg[y, x]) or value > agg[y, x]:
            agg[y, x] = value

    @staticmethod
    def _combine(aggs):
        raise NotImplementedError("last is not implemented for dask DataFrames")

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        return xr.DataArray(bases[0], **kwargs)



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
    def out_dshape(self, in_dshape, antialias):
        return dshape(Option(ct.float64))

    @staticmethod
    def _append(x, y, agg):
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

    Notes
    -----
    A single pass of the source dataset using antialiased lines can either be
    performed using a single-stage aggregation (e.g. ``self_intersect=True``)
    or two stages (``self_intersect=False``). If a ``summary`` contains a
    ``count`` or ``sum`` reduction with ``self_intersect=False``, or any of
    ``first``, ``last`` or ``min``, then the antialiased line pass will be
    performed in two stages.
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

    @property
    def inputs(self):
        return tuple(unique(concat(v.inputs for v in self.values)))


__all__ = list(set([_k for _k,_v in locals().items()
                    if isinstance(_v,type) and (issubclass(_v,Reduction) or _v is summary)
                    and _v not in [Reduction, OptionalFieldReduction,
                                   FloatingReduction, m2]])) + \
                    ['category_modulo', 'category_binning']
