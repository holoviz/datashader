from __future__ import annotations
import copy
from enum import Enum
from packaging.version import Version
import numpy as np
from datashape import dshape, isnumeric, Record, Option
from datashape import coretypes as ct
from toolz import concat, unique
import xarray as xr

from datashader.antialias import AntialiasCombination, AntialiasStage2
from datashader.utils import isminus1, isnull
from numba import cuda as nb_cuda

try:
    from datashader.transfer_functions._cuda_utils import (
        cuda_atomic_nanmin, cuda_atomic_nanmax, cuda_args, cuda_row_min_in_place,
        cuda_nanmax_n_in_place_4d, cuda_nanmax_n_in_place_3d,
        cuda_nanmin_n_in_place_4d, cuda_nanmin_n_in_place_3d,
        cuda_row_max_n_in_place_4d, cuda_row_max_n_in_place_3d,
        cuda_row_min_n_in_place_4d, cuda_row_min_n_in_place_3d, cuda_shift_and_insert,
    )
except ImportError:
    (cuda_atomic_nanmin, cuda_atomic_nanmax, cuda_args, cuda_row_min_in_place,
        cuda_nanmax_n_in_place_4d, cuda_nanmax_n_in_place_3d,
        cuda_nanmin_n_in_place_4d, cuda_nanmin_n_in_place_3d,
        cuda_row_max_n_in_place_4d, cuda_row_max_n_in_place_3d,
        cuda_row_min_n_in_place_4d, cuda_row_min_n_in_place_3d, cuda_shift_and_insert,
    ) = None, None, None, None, None, None, None, None, None, None, None, None, None

try:
    import cudf
    import cupy as cp
except Exception:
    cudf = cp = None

from .utils import (
    Expr, ngjit, nansum_missing, nanmax_in_place, nansum_in_place, row_min_in_place,
    nanmax_n_in_place_4d, nanmax_n_in_place_3d, nanmin_n_in_place_4d, nanmin_n_in_place_3d,
    row_max_n_in_place_4d, row_max_n_in_place_3d, row_min_n_in_place_4d, row_min_n_in_place_3d,
    shift_and_insert,
)


class SpecialColumn(Enum):
    """
    Internally datashader identifies the columns required by the user's
    Reductions and extracts them from the supplied source (e.g. DataFrame) to
    pass through the dynamically-generated append function in compiler.py and
    end up as arguments to the Reduction._append* functions. Each column is
    a string name or a SpecialColumn. A column of None is used in Reduction
    classes to denote that no column is required.
    """
    RowIndex = 1


class UsesCudaMutex(Enum):
    """
    Enum that encapsulates the need for a Reduction to use a CUDA mutex to
    operate correctly on a GPU. Possible values:

    No: the Reduction append_cuda function is atomic and no mutex is required.
    Local: Reduction append_cuda needs wrapping in a mutex.
    Global: the overall compiled append function needs wrapping in a mutex.
    """
    No = 0
    Local = 1
    Global = 2


class Preprocess(Expr):
    """Base clase for preprocessing steps."""
    def __init__(self, column: str | SpecialColumn | None):
        self.column = column

    @property
    def inputs(self):
        return (self.column,)

    @property
    def nan_check_column(self):
        return None


class extract(Preprocess):
    """Extract a column from a dataframe as a numpy array of values."""
    def apply(self, df):
        if self.column is SpecialColumn.RowIndex:
            attrs = getattr(df, "attrs", None)
            row_offset = getattr(attrs or df, "_datashader_row_offset", 0)

        if cudf and isinstance(df, cudf.DataFrame):
            if self.column is SpecialColumn.RowIndex:
                return cp.arange(row_offset, row_offset+len(df), dtype=np.int64)

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
        elif self.column is SpecialColumn.RowIndex:
            return np.arange(row_offset, row_offset+len(df), dtype=np.int64)
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

        index_float = (values - self.bin0) / self.binsize
        # NaN values are corrected below, so set them to zero to avoid warnings when
        # converting from float to int.
        index_float[nan_values] = 0
        index = index_float.astype(int)
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
            if self.column == SpecialColumn.RowIndex:
                nullval = -1
            elif df[self.column].dtype.kind == 'f':
                nullval = np.nan
            else:
                nullval = 0
            a = cupy.asarray(a)
            if self.column == SpecialColumn.RowIndex:
                b = extract(SpecialColumn.RowIndex).apply(df)
            elif Version(cudf.__version__) >= Version("22.02"):
                b = df[self.column].to_cupy(na_value=nullval)
            else:
                b = cupy.asarray(df[self.column].fillna(nullval))
            return cupy.stack((a, b), axis=-1)
        else:
            if self.column == SpecialColumn.RowIndex:
                b = extract(SpecialColumn.RowIndex).apply(df)
            else:
                b = df[self.column].values
            return np.stack((a, b), axis=-1)


class Reduction(Expr):
    """Base class for per-bin reductions."""
    def __init__(self, column: str | SpecialColumn | None=None):
        self.column = column
        self._nan_check_column = None

    @property
    def nan_check_column(self):
        if self._nan_check_column is not None:
            return extract(self._nan_check_column)
        else:
            return None

    def uses_cuda_mutex(self) -> UsesCudaMutex:
        """Return ``True`` if this Reduction needs to use a CUDA mutex to
        ensure that it is threadsafe across CUDA threads.

        If the CUDA append functions are all atomic (i.e. using functions from
        the numba.cuda.atomic module) then this is ``False``, otherwise it is
        ``True``.
        """
        return UsesCudaMutex.No

    def uses_row_index(self, cuda, partitioned):
        """Return ``True`` if this Reduction uses a row index virtual column.

        For some reductions the order of the rows of supplied data is
        important. These include ``first`` and ``last`` reductions as well as
        ``where`` reductions that return a row index. In some situations the
        order is intrinsic such as ``first`` reductions that are processed
        sequentially (i.e. on a CPU without using Dask) and no extra column is
        required. But in situations of parallel processing (using a GPU or
        Dask) extra information is needed that is provided by a row index
        virtual column.

        Returning ``True`` from this function will cause a row index column to
        be created and passed to the ``append`` functions in the usual manner.
        """
        return False

    def validate(self, in_dshape):
        if self.column == SpecialColumn.RowIndex:
            return
        if not self.column in in_dshape.dict:
            raise ValueError("specified column not found")
        if not isnumeric(in_dshape.measure[self.column]):
            raise ValueError("input must be numeric")

    @property
    def inputs(self):
        return (extract(self.column),)

    def is_categorical(self):
        """Return ``True`` if this is or contains a categorical reduction."""
        return False

    def is_where(self):
        """Return ``True`` if this is a ``where`` reduction or directly wraps
        a where reduction."""
        return False

    def _antialias_requires_2_stages(self):
        # Return True if this Reduction must be processed with 2 stages,
        # False if it doesn't matter.
        # Overridden in derived classes as appropriate.
        return False

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        # Only called if using antialiased lines. Overridden in derived classes.
        # Returns a tuple containing an item for each constituent reduction.
        # Each item is (AntialiasCombination, zero_value)).
        raise NotImplementedError(f"{type(self)}._antialias_stage_2 is not defined")

    def _build_bases(self, cuda, partitioned):
        return (self,)

    def _build_combine_temps(self, cuda, partitioned):
        # Temporaries (i.e. not returned to user) that are reductions, the
        # aggs of which are passed to the combine() function but not the
        # append() functions, as opposed to _build_temps() which are passed
        # to both append() and combine().
        return ()

    def _build_temps(self, cuda=False):
        # Temporaries (i.e. not returned to user) that are reductions, the
        # aggs of which are passed to both append() and combine() functions.
        return ()

    def _build_create(self, required_dshape):
        fields = getattr(required_dshape.measure, "fields", None)
        if fields is not None and len(required_dshape.measure.fields) > 0:
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
        elif required_dshape == dshape(ct.int64):
            return self._create_int64
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

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical = False):
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
    def _create_int64(shape, array_module):
        return array_module.full(shape, -1, dtype='i8')

    @staticmethod
    def _create_uint32(shape, array_module):
        return array_module.zeros(shape, dtype='u4')


class OptionalFieldReduction(Reduction):
    """Base class for things like ``count`` or ``any`` for which the field is optional"""
    def __init__(self, column=None):
        super().__init__(column)

    @property
    def inputs(self):
        return (extract(self.column),) if self.column is not None else ()

    def validate(self, in_dshape):
        if self.column is not None:
            super().validate(in_dshape)

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
    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
        return dshape(ct.float32) if antialias else dshape(ct.uint32)

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        if self_intersect:
            return (AntialiasStage2(AntialiasCombination.SUM_1AGG, array_module.nan),)
        else:
            return (AntialiasStage2(AntialiasCombination.SUM_2AGG, array_module.nan),)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            agg[y, x] += 1
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        if not isnull(field):
            if isnull(agg[y, x]):
                agg[y, x] = aa_factor
            else:
                agg[y, x] += aa_factor
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias_not_self_intersect(x, y, agg, field, aa_factor):
        if not isnull(field):
            if isnull(agg[y, x]) or aa_factor > agg[y, x]:
                agg[y, x] = aa_factor
                return 0
        return -1

    @staticmethod
    @ngjit
    def _append_no_field(x, y, agg):
        agg[y, x] += 1
        return 0

    @staticmethod
    @ngjit
    def _append_no_field_antialias(x, y, agg, aa_factor):
        if isnull(agg[y, x]):
            agg[y, x] = aa_factor
        else:
            agg[y, x] += aa_factor
        return 0

    @staticmethod
    @ngjit
    def _append_no_field_antialias_not_self_intersect(x, y, agg, aa_factor):
        if isnull(agg[y, x]) or aa_factor > agg[y, x]:
            agg[y, x] = aa_factor
            return 0
        return -1

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_antialias_cuda(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value):
            old = cuda_atomic_nanmax(agg, (y, x), value)
            if isnull(old) or old < value:
                return 0
        return -1

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_no_field_antialias_cuda_not_self_intersect(x, y, agg, aa_factor):
        if not isnull(aa_factor):
            old = cuda_atomic_nanmax(agg, (y, x), aa_factor)
            if isnull(old) or old < aa_factor:
                return 0
        return -1

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if not isnull(field):
            nb_cuda.atomic.add(agg, (y, x), 1)
            return 0
        return -1

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_no_field_antialias_cuda(x, y, agg, aa_factor):
        if not isnull(aa_factor):
            old = cuda_atomic_nanmax(agg, (y, x), aa_factor)
            if isnull(old) or old < aa_factor:
                return 0
        return -1

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_no_field_cuda(x, y, agg):
        nb_cuda.atomic.add(agg, (y, x), 1)
        return 0

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical = False):
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
        super().__init__()

        # set basic categorizer
        if isinstance(cat_column, CategoryPreprocess):
            self.categorizer = cat_column
        elif isinstance(cat_column, str):
            self.categorizer = category_codes(cat_column)
        else:
            raise TypeError("first argument must be a column name or a CategoryPreprocess instance")

        self.column = self.categorizer.column # for backwards compatibility with count_cat

        self.columns = (self.categorizer.column,)
        if (columns := getattr(reduction, 'columns', None)) is not None:
            # Must reverse columns (from where reduction) so that val_column property
            # is the column that is returned to the user.
            self.columns += columns[::-1]
        else:
            self.columns += (getattr(reduction, 'column', None),)

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

    def out_dshape(self, input_dshape, antialias, cuda, partitioned):
        cats = self.categorizer.categories(input_dshape)
        red_shape = self.reduction.out_dshape(input_dshape, antialias, cuda, partitioned)
        return dshape(Record([(c, red_shape) for c in cats]))

    @property
    def inputs(self):
        return (self.preprocess,)

    def is_categorical(self):
        return True

    def is_where(self):
        return self.reduction.is_where()

    @property
    def nan_check_column(self):
        return self.reduction.nan_check_column

    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return self.reduction.uses_cuda_mutex()

    def uses_row_index(self, cuda, partitioned):
        return self.reduction.uses_row_index(cuda, partitioned)

    def _antialias_requires_2_stages(self):
        return self.reduction._antialias_requires_2_stages()

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        ret = self.reduction._antialias_stage_2(self_intersect, array_module)
        return (AntialiasStage2(combination=ret[0].combination,
                                zero=ret[0].zero,
                                n_reduction=ret[0].n_reduction,
                                categorical=True),)

    def _build_create(self, required_dshape):
        n_cats = len(required_dshape.measure.fields)
        return lambda shape, array_module: self.reduction._build_create(
            required_dshape)(shape + (n_cats,), array_module)

    def _build_bases(self, cuda, partitioned):
        bases = self.reduction._build_bases(cuda, partitioned)
        if len(bases) == 1 and bases[0] is self:
            return bases
        return tuple(by(self.categorizer, base) for base in bases)

    def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
        return self.reduction._build_append(dshape, schema, cuda, antialias, self_intersect)

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical = False):
        return self.reduction._build_combine(dshape, antialias, cuda, partitioned, True)

    def _build_combine_temps(self, cuda, partitioned):
        return self.reduction._build_combine_temps(cuda, partitioned)

    def _build_finalize(self, dshape):
        cats = list(self.categorizer.categories(dshape))

        def finalize(bases, cuda=False, **kwargs):
            # Return a modified copy of kwargs. Cannot modify supplied kwargs as it
            # may be used by multiple reductions, e.g. if a summary reduction.
            kwargs = copy.deepcopy(kwargs)
            kwargs['dims'] += [self.cat_column]
            kwargs['coords'][self.cat_column] = cats
            return self.reduction._build_finalize(dshape)(bases, cuda=cuda, **kwargs)

        return finalize

class any(OptionalFieldReduction):
    """Whether any elements in ``column`` map to each bin.

    Parameters
    ----------
    column : str, optional
        If provided, any elements in ``column`` that are ``NaN`` are skipped.
    """
    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
        return dshape(ct.float32) if antialias else dshape(ct.bool_)

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MAX, array_module.nan),)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            agg[y, x] = True
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        if not isnull(field):
            if isnull(agg[y, x]) or aa_factor > agg[y, x]:
                agg[y, x] = aa_factor
                return 0
        return -1

    @staticmethod
    @ngjit
    def _append_no_field(x, y, agg):
        agg[y, x] = True
        return 0

    @staticmethod
    @ngjit
    def _append_no_field_antialias(x, y, agg, aa_factor):
        if isnull(agg[y, x]) or aa_factor > agg[y, x]:
            agg[y, x] = aa_factor
            return 0
        return -1

    # GPU append functions
    _append_cuda =_append
    _append_no_field_cuda = _append_no_field

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical = False):
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
    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
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
    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
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
    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        if self_intersect:
            return (AntialiasStage2(AntialiasCombination.SUM_1AGG, 0),)
        else:
            return (AntialiasStage2(AntialiasCombination.SUM_2AGG, 0),)

    def _build_create(self, required_dshape):
        return self._create_float64_zero

    # CPU append functions.
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            # agg[y, x] cannot be null as initialised to zero.
            agg[y, x] += field
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value):
            # agg[y, x] cannot be null as initialised to zero.
            agg[y, x] += value
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias_not_self_intersect(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value) and value > agg[y, x]:
            # agg[y, x] cannot be null as initialised to zero.
            agg[y, x] = value
            return 0
        return -1

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if not isnull(field):
            nb_cuda.atomic.add(agg, (y, x), field)
            return 0
        return -1

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
    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        if self_intersect:
            return (AntialiasStage2(AntialiasCombination.SUM_1AGG, array_module.nan),)
        else:
            return (AntialiasStage2(AntialiasCombination.SUM_2AGG, array_module.nan),)

    def _build_bases(self, cuda, partitioned):
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
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value):
            if isnull(agg[y, x]):
                agg[y, x] = value
            else:
                agg[y, x] += value
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias_not_self_intersect(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value):
            if isnull(agg[y, x]) or value > agg[y, x]:
                agg[y, x] = value
                return 0
        return -1

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
    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return UsesCudaMutex.Global

    def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
        return super(m2, self)._build_append(dshape, schema, cuda, antialias, self_intersect)

    def _build_create(self, required_dshape):
        return self._create_float64_zero

    def _build_temps(self, cuda=False):
        return (_sum_zero(self.column), count(self.column))

    # CPU append functions
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
                return 0
        return -1

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, m2, field, sum, count):
        # sum & count are the results of sum[y, x], count[y, x] before being
        # updated by field
        if not isnull(field):
            if count > 0:
                u1 = np.float64(sum) / count
                u = np.float64(sum + field) / (count + 1)
                m2[y, x] += (field - u1) * (field - u)
                return 0
        return -1

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

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MIN, array_module.nan),)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field) and (isnull(agg[y, x]) or agg[y, x] > field):
            agg[y, x] = field
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value) and (isnull(agg[y, x]) or value > agg[y, x]):
            agg[y, x] = value
            return 0
        return -1

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if not isnull(field):
            old = cuda_atomic_nanmin(agg, (y, x), field)
            if isnull(old) or old > field:
                return 0
        return -1

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
    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MAX, array_module.nan),)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field) and (isnull(agg[y, x]) or agg[y, x] < field):
            agg[y, x] = field
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value) and (isnull(agg[y, x]) or value > agg[y, x]):
            agg[y, x] = value
            return 0
        return -1

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_antialias_cuda(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value):
            old = cuda_atomic_nanmax(agg, (y, x), value)
            if isnull(old) or old < value:
                return 0
        return -1

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if not isnull(field):
            old = cuda_atomic_nanmax(agg, (y, x), field)
            if isnull(old) or old < field:
                return 0
        return -1

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
    def _build_bases(self, cuda, partitioned):
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
    def _build_bases(self, cuda, partitioned):
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
    def _build_bases(self, cuda, partitioned):
        return (_sum_zero(self.column), count(self.column), m2(self.column))

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        sums, counts, m2s = bases
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.where(counts > 0, np.sqrt(m2s / counts), np.nan)
        return xr.DataArray(x, **kwargs)


class _first_or_last(Reduction):
    """Abstract base class of first and last reductions.
    """
    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
        return dshape(ct.float64)

    def uses_row_index(self, cuda, partitioned):
        return cuda or partitioned

    def _antialias_requires_2_stages(self):
        return True

    def _build_bases(self, cuda, partitioned):
        if self.uses_row_index(cuda, partitioned):
            row_index_selector = self._create_row_index_selector()
            wrapper = where(selector=row_index_selector, lookup_column=self.column)
            wrapper._nan_check_column = self.column
            # where reduction is always preceded by its selector reduction
            return row_index_selector._build_bases(cuda, partitioned) + (wrapper,)
        else:
            return super()._build_bases(cuda, partitioned)

    @staticmethod
    def _combine(aggs):
        # Dask combine is handled by a where reduction using a row index.
        # Hence this can only ever be called if npartitions == 1 in which case len(aggs) == 1.
        if len(aggs) > 1:
            raise RuntimeError("_combine should never be called with more than one agg")
        return aggs[0]

    def _create_row_index_selector(self):
        pass

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        # Note returning the last of the bases which is correct regardless of whether
        # this is a simple reduction (with a single base) or a compound where reduction
        # (with 2 bases, the second of which is the where reduction).
        return xr.DataArray(bases[-1], **kwargs)


class first(_first_or_last):
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
    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.FIRST, array_module.nan),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field) and isnull(agg[y, x]):
            agg[y, x] = field
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value) and (isnull(agg[y, x]) or value > agg[y, x]):
            agg[y, x] = value
            return 0
        return -1

    def _create_row_index_selector(self):
        return _min_row_index()


class last(_first_or_last):
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
    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.LAST, array_module.nan),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            agg[y, x] = field
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value) and (isnull(agg[y, x]) or value > agg[y, x]):
            agg[y, x] = value
            return 0
        return -1

    def _create_row_index_selector(self):
        return _max_row_index()


class FloatingNReduction(OptionalFieldReduction):
    def __init__(self, column=None, n=1):
        super().__init__(column)
        self.n = n if n >= 1 else 1

    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
        return dshape(ct.float64)

    def _add_finalize_kwargs(self, **kwargs):
        # Add the new dimension and coordinate.
        n_name = "n"
        n_values = np.arange(self.n)

        # Return a modified copy of kwargs. Cannot modify supplied kwargs as it
        # may be used by multiple reductions, e.g. if a summary reduction.
        kwargs = copy.deepcopy(kwargs)
        kwargs['dims'] += [n_name]
        kwargs['coords'][n_name] = n_values
        return kwargs

    def _build_create(self, required_dshape):
        return lambda shape, array_module: super(FloatingNReduction, self)._build_create(
            required_dshape)(shape + (self.n,), array_module)

    def _build_finalize(self, dshape):
        def finalize(bases, cuda=False, **kwargs):
            kwargs = self._add_finalize_kwargs(**kwargs)
            return self._finalize(bases, cuda=cuda, **kwargs)

        return finalize

    def _hashable_inputs(self):
        return super()._hashable_inputs() + (self.n,)


class _first_n_or_last_n(FloatingNReduction):
    """Abstract base class of first_n and last_n reductions.
    """
    def uses_row_index(self, cuda, partitioned):
        return cuda or partitioned

    def _antialias_requires_2_stages(self):
        return True

    def _build_bases(self, cuda, partitioned):
        if self.uses_row_index(cuda, partitioned):
            row_index_selector = self._create_row_index_selector()
            wrapper = where(selector=row_index_selector, lookup_column=self.column)
            wrapper._nan_check_column = self.column
            # where reduction is always preceded by its selector reduction
            return row_index_selector._build_bases(cuda, partitioned) + (wrapper,)
        else:
            return super()._build_bases(cuda, partitioned)

    @staticmethod
    def _combine(aggs):
        # Dask combine is handled by a where reduction using a row index.
        # Hence this can only ever be called if npartitions == 1 in which case len(aggs) == 1.
        if len(aggs) > 1:
            raise RuntimeError("_combine should never be called with more than one agg")
        return aggs[0]

    def _create_row_index_selector(self):
        pass

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        # Note returning the last of the bases which is correct regardless of whether
        # this is a simple reduction (with a single base) or a compound where reduction
        # (with 2 bases, the second of which is the where reduction).
        return xr.DataArray(bases[-1], **kwargs)


class first_n(_first_n_or_last_n):
    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.FIRST, array_module.nan, n_reduction=True),)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            # Check final value first for quick abort.
            n = agg.shape[2]
            if not isnull(agg[y, x, n-1]):
                return -1

            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            for i in range(n):
                if isnull(agg[y, x, i]):
                    # Nothing to shift.
                    agg[y, x, i] = field
                    return i
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value):
            # Check final value first for quick abort.
            n = agg.shape[2]
            if not isnull(agg[y, x, n-1]):
                return -1

            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            for i in range(n):
                if isnull(agg[y, x, i]):
                    # Nothing to shift.
                    agg[y, x, i] = value
                    return i
        return -1

    def _create_row_index_selector(self):
        return _min_n_row_index(n=self.n)


class last_n(_first_n_or_last_n):
    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.LAST, array_module.nan, n_reduction=True),)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            # Always inserts at front of agg's third dimension.
            shift_and_insert(agg[y, x], field, 0)
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value):
            # Always inserts at front of agg's third dimension.
            shift_and_insert(agg[y, x], value, 0)
            return 0
        return -1

    def _create_row_index_selector(self):
        return _max_n_row_index(n=self.n)


class max_n(FloatingNReduction):
    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return UsesCudaMutex.Local

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MAX, array_module.nan, n_reduction=True),)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            n = agg.shape[2]
            for i in range(n):
                if isnull(agg[y, x, i]) or field > agg[y, x, i]:
                    shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value):
            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            n = agg.shape[2]
            for i in range(n):
                if isnull(agg[y, x, i]) or value > agg[y, x, i]:
                    shift_and_insert(agg[y, x], value, i)
                    return i
        return -1

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if not isnull(field):
            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            n = agg.shape[2]
            for i in range(n):
                if isnull(agg[y, x, i]) or field > agg[y, x, i]:
                    cuda_shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical = False):
        if cuda:
            return self._combine_cuda
        else:
            return self._combine

    @staticmethod
    def _combine(aggs):
        ret = aggs[0]
        for i in range(1, len(aggs)):
            if ret.ndim == 3:  # ndim is either 3 (ny, nx, n) or 4 (ny, nx, ncat, n)
                nanmax_n_in_place_3d(aggs[0], aggs[i])
            else:
                nanmax_n_in_place_4d(aggs[0], aggs[i])
        return ret

    @staticmethod
    def _combine_cuda(aggs):
        ret = aggs[0]
        kernel_args = cuda_args(ret.shape[:-1])
        for i in range(1, len(aggs)):
            if ret.ndim == 3:  # ndim is either 3 (ny, nx, n) or 4 (ny, nx, ncat, n)
                cuda_nanmax_n_in_place_3d[kernel_args](aggs[0], aggs[i])
            else:
                cuda_nanmax_n_in_place_4d[kernel_args](aggs[0], aggs[i])
        return ret


class min_n(FloatingNReduction):
    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return UsesCudaMutex.Local

    def _antialias_requires_2_stages(self):
        return True

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MIN, array_module.nan, n_reduction=True),)

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            n = agg.shape[2]
            for i in range(n):
                if isnull(agg[y, x, i]) or field < agg[y, x, i]:
                    shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        value = field*aa_factor
        if not isnull(value):
            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            n = agg.shape[2]
            for i in range(n):
                if isnull(agg[y, x, i]) or value < agg[y, x, i]:
                    shift_and_insert(agg[y, x], value, i)
                    return i
        return -1

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if not isnull(field):
            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            n = agg.shape[2]
            for i in range(n):
                if isnull(agg[y, x, i]) or field < agg[y, x, i]:
                    cuda_shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical = False):
        if cuda:
            return self._combine_cuda
        else:
            return self._combine

    @staticmethod
    def _combine(aggs):
        ret = aggs[0]
        for i in range(1, len(aggs)):
            if ret.ndim == 3:  # ndim is either 3 (ny, nx, n) or 4 (ny, nx, ncat, n)
                nanmin_n_in_place_3d(aggs[0], aggs[i])
            else:
                nanmin_n_in_place_4d(aggs[0], aggs[i])
        return ret

    @staticmethod
    def _combine_cuda(aggs):
        ret = aggs[0]
        kernel_args = cuda_args(ret.shape[:-1])
        for i in range(1, len(aggs)):
            if ret.ndim == 3:  # ndim is either 3 (ny, nx, n) or 4 (ny, nx, ncat, n)
                cuda_nanmin_n_in_place_3d[kernel_args](aggs[0], aggs[i])
            else:
                cuda_nanmin_n_in_place_4d[kernel_args](aggs[0], aggs[i])
        return ret


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
    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
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


class where(FloatingReduction):
    """
    Returns values from a ``lookup_column`` corresponding to a ``selector``
    reduction that is applied to some other column.

    If ``lookup_column`` is ``None`` then it uses the index of the row in the
    DataFrame instead of a named column. This is returned as an int64
    aggregation with -1 used to denote no value.

    Examples
    --------
    >>> canvas.line(df, 'x', 'y', agg=ds.where(ds.max("value"), "other"))  # doctest: +SKIP

    This returns the values of the "other" column that correspond to the
    maximum of the "value" column in each bin.

    Parameters
    ----------
    selector: Reduction
        Reduction used to select the values of the ``lookup_column`` which are
        returned by this ``where`` reduction.

    lookup_column : str | None
        Column containing values that are returned from this ``where``
        reduction, or ``None`` to return row indexes instead.
    """
    def __init__(self, selector: Reduction, lookup_column: str | None=None):
        if not isinstance(selector, (first, first_n, last, last_n, max, max_n, min, min_n,
                                     _max_or_min_row_index, _max_n_or_min_n_row_index)):
            raise TypeError(
                "selector can only be a first, first_n, last, last_n, "
                "max, max_n, min or min_n reduction")
        if lookup_column is None:
            lookup_column = SpecialColumn.RowIndex
        super().__init__(lookup_column)
        self.selector = selector
        # List of all column names that this reduction uses.
        self.columns = (selector.column, lookup_column)

    def __hash__(self):
        return hash((type(self), self._hashable_inputs(), self.selector))

    def is_where(self):
        return True

    def out_dshape(self, input_dshape, antialias, cuda, partitioned):
        if self.column == SpecialColumn.RowIndex:
            return dshape(ct.int64)
        else:
            return dshape(ct.float64)

    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return UsesCudaMutex.Local

    def uses_row_index(self, cuda, partitioned):
        return self.column == SpecialColumn.RowIndex or self.selector.uses_row_index(cuda, partitioned)

    def validate(self, in_dshape):
        if self.column != SpecialColumn.RowIndex:
            super().validate(in_dshape)
        self.selector.validate(in_dshape)
        if self.column != SpecialColumn.RowIndex and self.column == self.selector.column:
            raise ValueError("where and its contained reduction cannot use the same column")

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        ret = self.selector._antialias_stage_2(self_intersect, array_module)
        if self.column == SpecialColumn.RowIndex:
            # Override antialiased zero value when returning integer row index.
            ret = (AntialiasStage2(combination=ret[0].combination,
                                   zero=-1,
                                   n_reduction=ret[0].n_reduction),)
        return ret

    # CPU append functions
    # All where._append* functions have an extra argument which is the update index.
    # For 3D aggs like max_n, this is the index of insertion in the final dimension,
    # and the previous values from this index upwards are shifted along to make room
    # for the new value.
    @staticmethod
    @ngjit
    def _append(x, y, agg, field, update_index):
        if agg.ndim > 2:
            shift_and_insert(agg[y, x], field, update_index)
        else:
            agg[y, x] = field
        return update_index

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor, update_index):
        # Ignore aa_factor.
        if agg.ndim > 2:
            shift_and_insert(agg[y, x], field, update_index)
        else:
            agg[y, x] = field

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_antialias_cuda(x, y, agg, field, aa_factor, update_index):
        # Ignore aa_factor
        if agg.ndim > 2:
            cuda_shift_and_insert(agg[y, x], field, update_index)
        else:
            agg[y, x] = field
        return update_index

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field, update_index):
        if agg.ndim > 2:
            cuda_shift_and_insert(agg[y, x], field, update_index)
        else:
            agg[y, x] = field
        return update_index

    def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
        # If self.column is SpecialColumn.RowIndex then append function is passed a
        # 'field' argument which is the row index.
        if cuda:
            if antialias:
                return self._append_antialias_cuda
            else:
                return self._append_cuda
        else:
            if antialias:
                return self._append_antialias
            else:
                return self._append

    def _build_bases(self, cuda, partitioned):
        selector = self.selector
        if isinstance(selector, (_first_or_last, _first_n_or_last_n)) and selector.uses_row_index(cuda, partitioned):
            # Need to swap out the selector with an equivalent row index selector
            row_index_selector = selector._create_row_index_selector()
            if self.column == SpecialColumn.RowIndex:
                # If selector uses a row index and this where returns the same row index,
                # can just swap out this where reduction with the row_index_selector.
                row_index_selector._nan_check_column = self.selector.column
                return row_index_selector._build_bases(cuda, partitioned)
            else:
                new_where = where(row_index_selector, self.column)
                new_where._nan_check_column = self.selector.column
                return row_index_selector._build_bases(cuda, partitioned) + new_where._build_bases(cuda, partitioned)
        else:
            return selector._build_bases(cuda, partitioned) + super()._build_bases(cuda, partitioned)

    def _combine_callback(self, cuda, partitioned, categorical):
        # Used by:
        # 1) where._build_combine()) below, the usual mechanism for combining aggs from
        #    different dask partitions.
        # 2) make_antialias_stage_2_functions() in compiler.py to perform stage 2 combine
        #    of antialiased aggs.
        selector = self.selector
        is_n_reduction = isinstance(selector, FloatingNReduction)
        if cuda:
            append = selector._append_cuda
        else:
            append = selector._append

        # If the selector uses a row_index then selector_aggs will be int64 with -1
        # representing missing data. Otherwise missing data is NaN.
        invalid = isminus1 if self.selector.uses_row_index(cuda, partitioned) else isnull

        @ngjit
        def combine_cpu_2d(aggs, selector_aggs):
            ny, nx = aggs[0].shape
            for y in range(ny):
                for x in range(nx):
                    value = selector_aggs[1][y, x]
                    if not invalid(value) and append(x, y, selector_aggs[0], value) >= 0:
                        aggs[0][y, x] = aggs[1][y, x]

        @ngjit
        def combine_cpu_3d(aggs, selector_aggs):
            ny, nx, ncat = aggs[0].shape
            for y in range(ny):
                for x in range(nx):
                    for cat in range(ncat):
                        value = selector_aggs[1][y, x, cat]
                        if not invalid(value) and append(x, y, selector_aggs[0][:, :, cat], value) >= 0:
                            aggs[0][y, x, cat] = aggs[1][y, x, cat]

        @ngjit
        def combine_cpu_n_3d(aggs, selector_aggs):
            ny, nx, n = aggs[0].shape
            for y in range(ny):
                for x in range(nx):
                    for i in range(n):
                        value = selector_aggs[1][y, x, i]
                        if invalid(value):
                            break
                        update_index = append(x, y, selector_aggs[0], value)
                        if update_index < 0:
                            break
                        shift_and_insert(aggs[0][y, x], aggs[1][y, x, i], update_index)

        @ngjit
        def combine_cpu_n_4d(aggs, selector_aggs):
            ny, nx, ncat, n = aggs[0].shape
            for y in range(ny):
                for x in range(nx):
                    for cat in range(ncat):
                        for i in range(n):
                            value = selector_aggs[1][y, x, cat, i]
                            if invalid(value):
                                break
                            update_index = append(x, y, selector_aggs[0][:, :, cat, :], value)
                            if update_index < 0:
                                break
                            shift_and_insert(aggs[0][y, x, cat], aggs[1][y, x, cat, i], update_index)

        @nb_cuda.jit
        def combine_cuda_2d(aggs, selector_aggs):
            ny, nx = aggs[0].shape
            x, y = nb_cuda.grid(2)
            if x < nx and y < ny:
                value = selector_aggs[1][y, x]
                if not invalid(value) and append(x, y, selector_aggs[0], value) >= 0:
                    aggs[0][y, x] = aggs[1][y, x]

        @nb_cuda.jit
        def combine_cuda_3d(aggs, selector_aggs):
            ny, nx, ncat = aggs[0].shape
            x, y, cat = nb_cuda.grid(3)
            if x < nx and y < ny and cat < ncat:
                value = selector_aggs[1][y, x, cat]
                if not invalid(value) and append(x, y, selector_aggs[0][:, :, cat], value) >= 0:
                    aggs[0][y, x, cat] = aggs[1][y, x, cat]

        @nb_cuda.jit
        def combine_cuda_n_3d(aggs, selector_aggs):
            ny, nx, n = aggs[0].shape
            x, y = nb_cuda.grid(2)
            if x < nx and y < ny:
                for i in range(n):
                    value = selector_aggs[1][y, x, i]
                    if invalid(value):
                        break
                    update_index = append(x, y, selector_aggs[0], value)
                    if update_index < 0:
                        break
                    cuda_shift_and_insert(aggs[0][y, x], aggs[1][y, x, i], update_index)

        @nb_cuda.jit
        def combine_cuda_n_4d(aggs, selector_aggs):
            ny, nx, ncat, n = aggs[0].shape
            x, y, cat = nb_cuda.grid(3)
            if x < nx and y < ny and cat < ncat:
                for i in range(n):
                    value = selector_aggs[1][y, x, cat, i]
                    if invalid(value):
                        break
                    update_index = append(x, y, selector_aggs[0][:, :, cat, :], value)
                    if update_index < 0:
                        break
                    cuda_shift_and_insert(aggs[0][y, x, cat], aggs[1][y, x, cat, i], update_index)

        if is_n_reduction:
            # ndim is either 3 (ny, nx, n) or 4 (ny, nx, ncat, n)
            if cuda:
                return combine_cuda_n_4d if categorical else combine_cuda_n_3d
            else:
                return combine_cpu_n_4d if categorical else combine_cpu_n_3d
        else:
            # ndim is either 2 (ny, nx) or 3 (ny, nx, ncat)
            if cuda:
                return combine_cuda_3d if categorical else combine_cuda_2d
            else:
                return combine_cpu_3d if categorical else combine_cpu_2d

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical = False):
        combine = self._combine_callback(cuda, partitioned, categorical)

        def wrapped_combine(aggs, selector_aggs):
            if len(aggs) == 1:
                pass
            elif cuda:
                is_n_reduction = isinstance(self.selector, FloatingNReduction)
                shape = aggs[0].shape[:-1] if is_n_reduction else aggs[0].shape
                combine[cuda_args(shape)](aggs, selector_aggs)
            else:
                combine(aggs, selector_aggs)

            return aggs[0], selector_aggs[0]

        return wrapped_combine

    def _build_combine_temps(self, cuda, partitioned):
        return (self.selector,)

    def _build_create(self, required_dshape):
        # Return a function that when called with a shape creates an agg array
        # of the required type (numpy/cupy) and dtype.
        if isinstance(self.selector, FloatingNReduction):
            # This specialisation isn't ideal but Reduction classes do not
            # store information about the required extra dimension.
            return lambda shape, array_module: super(where, self)._build_create(
                required_dshape)(shape + (self.selector.n,), array_module)
        else:
            return super()._build_create(required_dshape)

    def _build_finalize(self, dshape):
        if isinstance(self.selector, FloatingNReduction):
            add_finalize_kwargs = self.selector._add_finalize_kwargs
        else:
            add_finalize_kwargs = None

        def finalize(bases, cuda=False, **kwargs):
            if add_finalize_kwargs is not None:
                kwargs = add_finalize_kwargs(**kwargs)

            return xr.DataArray(bases[-1], **kwargs)

        return finalize


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

    def is_categorical(self):
        for v in self.values:
            if v.is_categorical():
                return True
        return False

    def uses_row_index(self, cuda, partitioned):
        for v in self.values:
            if v.uses_row_index(cuda, partitioned):
                return True
        return False

    def validate(self, input_dshape):
        for v in self.values:
            v.validate(input_dshape)

        # Check that any included FloatingNReductions have the same n values.
        n_values = []
        for v in self.values:
            if isinstance(v, where):
                v = v.selector
            if isinstance(v, FloatingNReduction):
                n_values.append(v.n)
        if len(np.unique(n_values)) > 1:
            raise ValueError(
                "Using multiple FloatingNReductions with different n values is not supported")

    @property
    def inputs(self):
        return tuple(unique(concat(v.inputs for v in self.values)))


class _max_or_min_row_index(OptionalFieldReduction):
    """Abstract base class of max and min row_index reductions.
    """
    def __init__(self):
        super().__init__(column=SpecialColumn.RowIndex)

    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
        return dshape(ct.int64)

    def uses_row_index(self, cuda, partitioned):
        return True


class _max_row_index(_max_or_min_row_index):
    """Max reduction operating on row index.

    This is a private class as it is not intended to be used explicitly in
    user code. It is primarily purpose is to support the use of ``last``
    reductions using dask and/or CUDA.
    """
    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MAX, -1),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        # field is int64 row index
        if field > agg[y, x]:
            agg[y, x] = field
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        # field is int64 row index
        # Ignore aa_factor
        if field > agg[y, x]:
            agg[y, x] = field
            return 0
        return -1

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        # field is int64 row index
        if field != -1:
            old = nb_cuda.atomic.max(agg, (y, x), field)
            if old < field:
                return 0
        return -1

    @staticmethod
    def _combine(aggs):
        # Maximum ignoring -1 values
        # Works for CPU and GPU
        if len(aggs) > 1:
            # Works with numpy or cupy arrays
            np.maximum(aggs[0], aggs[1], out=aggs[0])
        return aggs[0]


class _min_row_index(_max_or_min_row_index):
    """Min reduction operating on row index.

    This is a private class as it is not intended to be used explicitly in
    user code. It is primarily purpose is to support the use of ``first``
    reductions using dask and/or CUDA.
    """
    def _antialias_requires_2_stages(self):
        return True

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MIN, -1),)

    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return UsesCudaMutex.Local

    # CPU append functions
    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        # field is int64 row index
        if field != -1 and (agg[y, x] == -1 or field < agg[y, x]):
            agg[y, x] = field
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        # field is int64 row index
        # Ignore aa_factor
        if field != -1 and (agg[y, x] == -1 or field < agg[y, x]):
            agg[y, x] = field
            return 0
        return -1

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        # field is int64 row index
        # Always uses cuda mutex so this does not need to be atomic
        if field != -1 and (agg[y, x] == -1 or field < agg[y, x]):
            agg[y, x] = field
            return 0
        return -1

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical = False):
        if cuda:
            return self._combine_cuda
        else:
            return self._combine

    @staticmethod
    def _combine(aggs):
        # Minimum ignoring -1 values
        ret = aggs[0]
        if len(aggs) > 1:
            # Can take 2d (ny, nx) or 3d (ny, nx, ncat) arrays.
            row_min_in_place(aggs[0], aggs[1])
        return ret

    @staticmethod
    def _combine_cuda(aggs):
        ret = aggs[0]
        if len(aggs) > 1:
            if ret.ndim == 2:  # ndim is either 2 (ny, nx) or 3 (ny, nx, ncat)
                # 3d view of each agg
                aggs = [cp.expand_dims(agg, 2) for agg in aggs]
            kernel_args = cuda_args(ret.shape[:3])
            for i in range(1, len(aggs)):
                cuda_row_min_in_place[kernel_args](aggs[0], aggs[i])
        return ret


class _max_n_or_min_n_row_index(FloatingNReduction):
    """Abstract base class of max_n and min_n row_index reductions.
    """
    def __init__(self, n=1):
        super().__init__(column=SpecialColumn.RowIndex)
        self.n = n if n >= 1 else 1

    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
        return dshape(ct.int64)

    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return UsesCudaMutex.Local

    def uses_row_index(self, cuda, partitioned):
        return True

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical = False):
        if cuda:
            return self._combine_cuda
        else:
            return self._combine


class _max_n_row_index(_max_n_or_min_n_row_index):
    """Max_n reduction operating on row index.

    This is a private class as it is not intended to be used explicitly in
    user code. It is primarily purpose is to support the use of ``last_n``
    reductions using dask and/or CUDA.
    """
    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MAX, -1, n_reduction=True),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        # field is int64 row index
        if field != -1:
            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            n = agg.shape[2]
            for i in range(n):
                if agg[y, x, i] == -1 or field > agg[y, x, i]:
                    shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        # field is int64 row index
        # Ignoring aa_factor
        if field != -1:
            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            n = agg.shape[2]
            for i in range(n):
                if agg[y, x, i] == -1 or field > agg[y, x, i]:
                    # Bump previous values along to make room for new value.
                    for j in range(n-1, i, -1):
                        agg[y, x, j] = agg[y, x, j-1]
                    agg[y, x, i] = field
                    return i
        return -1

    # GPU append functions
    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        # field is int64 row index
        # Always uses cuda mutex so this does not need to be atomic
        if field != -1:
            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            n = agg.shape[2]
            for i in range(n):
                if agg[y, x, i] == -1 or field > agg[y, x, i]:
                    cuda_shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    @staticmethod
    def _combine(aggs):
        ret = aggs[0]
        if len(aggs) > 1:
            if ret.ndim == 3:  # ndim is either 3 (ny, nx, n) or 4 (ny, nx, ncat, n)
                row_max_n_in_place_3d(aggs[0], aggs[1])
            else:
                row_max_n_in_place_4d(aggs[0], aggs[1])
        return ret

    @staticmethod
    def _combine_cuda(aggs):
        ret = aggs[0]
        if len(aggs) > 1:
            kernel_args = cuda_args(ret.shape[:-1])
            if ret.ndim == 3:  # ndim is either 3 (ny, nx, n) or 4 (ny, nx, ncat, n)
                cuda_row_max_n_in_place_3d[kernel_args](aggs[0], aggs[1])
            else:
                cuda_row_max_n_in_place_4d[kernel_args](aggs[0], aggs[1])
        return ret


class _min_n_row_index(_max_n_or_min_n_row_index):
    """Min_n reduction operating on row index.

    This is a private class as it is not intended to be used explicitly in
    user code. It is primarily purpose is to support the use of ``first_n``
    reductions using dask and/or CUDA.
    """
    def _antialias_requires_2_stages(self):
        return True

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MIN, -1, n_reduction=True),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        # field is int64 row index
        if field != -1:
            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            n = agg.shape[2]
            for i in range(n):
                if agg[y, x, i] == -1 or field < agg[y, x, i]:
                    shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor):
        # field is int64 row index
        # Ignoring aa_factor
        if field != -1:
            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            n = agg.shape[2]
            for i in range(n):
                if agg[y, x, i] == -1 or field < agg[y, x, i]:
                    shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        # field is int64 row index
        # Always uses cuda mutex so this does not need to be atomic
        if field != -1:
            # Linear walk along stored values.
            # Could do binary search instead but not expecting n to be large.
            n = agg.shape[2]
            for i in range(n):
                if agg[y, x, i] == -1 or field < agg[y, x, i]:
                    cuda_shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    @staticmethod
    def _combine(aggs):
        ret = aggs[0]
        if len(aggs) > 1:
            if ret.ndim == 3:  # ndim is either 3 (ny, nx, n) or 4 (ny, nx, ncat, n)
                row_min_n_in_place_3d(aggs[0], aggs[1])
            else:
                row_min_n_in_place_4d(aggs[0], aggs[1])
        return ret

    @staticmethod
    def _combine_cuda(aggs):
        ret = aggs[0]
        if len(aggs) > 1:
            kernel_args = cuda_args(ret.shape[:-1])
            if ret.ndim == 3:  # ndim is either 3 (ny, nx, n) or 4 (ny, nx, ncat, n)
                cuda_row_min_n_in_place_3d[kernel_args](aggs[0], aggs[1])
            else:
                cuda_row_min_n_in_place_4d[kernel_args](aggs[0], aggs[1])

        return ret


__all__ = list(set([_k for _k,_v in locals().items()
                    if isinstance(_v,type) and (issubclass(_v,Reduction) or _v is summary)
                    and _v not in [Reduction, OptionalFieldReduction,
                                   FloatingReduction, m2]])) + \
                    ['category_modulo', 'category_binning']
