from __future__ import annotations

import re

from functools import total_ordering
from packaging.version import Version

import numpy as np
import pandas as pd

from numba import jit
from pandas.api.extensions import (
    ExtensionDtype, ExtensionArray, register_extension_dtype)
from numbers import Integral

from pandas.api.types import pandas_dtype, is_extension_array_dtype


try:
    # See if we can register extension type with dask >= 1.1.0
    from dask.dataframe.extensions import make_array_nonempty
except ImportError:
    make_array_nonempty = None


def _validate_ragged_properties(start_indices, flat_array):
    """
    Validate that start_indices are flat_array arrays that may be used to
    represent a valid RaggedArray.

    Parameters
    ----------
    flat_array: numpy array containing concatenation
                of all nested arrays to be represented
                by this ragged array
    start_indices: unsiged integer numpy array the same
                   length as the ragged array where values
                   represent the index into flat_array where
                   the corresponding ragged array element
                   begins
    Raises
    ------
    ValueError:
        if input arguments are invalid or incompatible properties
    """

    # Validate start_indices
    if (not isinstance(start_indices, np.ndarray) or
            start_indices.dtype.kind != 'u' or
            start_indices.ndim != 1):
        raise ValueError("""
The start_indices property of a RaggedArray must be a 1D numpy array of
unsigned integers (start_indices.dtype.kind == 'u')
    Received value of type {typ}: {v}""".format(
            typ=type(start_indices), v=repr(start_indices)))

    # Validate flat_array
    if (not isinstance(flat_array, np.ndarray) or
            flat_array.ndim != 1):
        raise ValueError("""
The flat_array property of a RaggedArray must be a 1D numpy array
    Received value of type {typ}: {v}""".format(
            typ=type(flat_array), v=repr(flat_array)))

    # Validate start_indices values
    # We don't need to check start_indices < 0 because we already know that it
    # has an unsigned integer datatype
    #
    # Note that start_indices[i] == len(flat_array) is valid as it represents
    # and empty array element at the end of the ragged array.
    invalid_inds = start_indices > len(flat_array)

    if invalid_inds.any():
        some_invalid_vals = start_indices[invalid_inds[:10]]

        raise ValueError("""
Elements of start_indices must be less than the length of flat_array ({m})
    Invalid values include: {vals}""".format(
            m=len(flat_array), vals=repr(some_invalid_vals)))


# Internal ragged element array wrapper that provides
# equality, ordering, and hashing.
@total_ordering
class _RaggedElement(object):

    @staticmethod
    def ragged_or_nan(a):
        if np.isscalar(a) and np.isnan(a):
            return a
        else:
            return _RaggedElement(a)

    @staticmethod
    def array_or_nan(a):
        if np.isscalar(a) and np.isnan(a):
            return a
        else:
            return a.array

    def __init__(self, array):
        self.array = array

    def __hash__(self):
        return hash(self.array.tobytes())

    def __eq__(self, other):
        if not isinstance(other, _RaggedElement):
            return False
        return np.array_equal(self.array, other.array)

    def __lt__(self, other):
        if not isinstance(other, _RaggedElement):
            return NotImplemented
        return _lexograph_lt(self.array, other.array)

    def __repr__(self):
        array_repr = repr(self.array)
        return array_repr.replace('array', 'ragged_element')


@register_extension_dtype
class RaggedDtype(ExtensionDtype):
    """
    Pandas ExtensionDtype to represent a ragged array datatype

    Methods not otherwise documented here are inherited from ExtensionDtype;
    please see the corresponding method on that class for the docstring
    """
    type = np.ndarray
    base = np.dtype('O')
    _subtype_re = re.compile(r"^ragged\[(?P<subtype>\w+)\]$")
    _metadata = ('_dtype',)

    @property
    def name(self):
        return 'Ragged[{subtype}]'.format(subtype=self.subtype)

    def __repr__(self):
        return self.name

    @classmethod
    def construct_array_type(cls):
        return RaggedArray

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError("'construct_from_string' expects a string, got %s" % type(string))

        # lowercase string
        string = string.lower()

        msg = "Cannot construct a 'RaggedDtype' from '{}'"
        if string.startswith('ragged'):
            # Extract subtype
            try:
                subtype_string = cls._parse_subtype(string)
                return RaggedDtype(dtype=subtype_string)
            except Exception:
                raise TypeError(msg.format(string))
        else:
            raise TypeError(msg.format(string))

    def __init__(self, dtype=np.float64):
        if isinstance(dtype, RaggedDtype):
            self._dtype = dtype.subtype
        else:
            self._dtype = np.dtype(dtype)

    @property
    def subtype(self):
        return self._dtype

    @classmethod
    def _parse_subtype(cls, dtype_string):
        """
        Parse a datatype string to get the subtype

        Parameters
        ----------
        dtype_string: str
            A string like Ragged[subtype]

        Returns
        -------
        subtype: str

        Raises
        ------
        ValueError
            When the subtype cannot be extracted
        """
        # Be case insensitive
        dtype_string = dtype_string.lower()

        match = cls._subtype_re.match(dtype_string)
        if match:
            subtype_string = match.groupdict()['subtype']
        elif dtype_string == 'ragged':
            subtype_string = 'float64'
        else:
            raise ValueError("Cannot parse {dtype_string}".format(
                dtype_string=dtype_string))
        return subtype_string


def missing(v):
    return v is None or (np.isscalar(v) and np.isnan(v))


class RaggedArray(ExtensionArray):
    """
    Pandas ExtensionArray to represent ragged arrays

    Methods not otherwise documented here are inherited from ExtensionArray;
    please see the corresponding method on that class for the docstring
    """
    def __init__(self, data, dtype=None, copy=False):
        """
        Construct a RaggedArray

        Parameters
        ----------
        data: list or array or dict or RaggedArray
            * list or 1D-array: A List or 1D array of lists or 1D arrays that
                                should be represented by the RaggedArray

            * dict: A dict containing 'start_indices' and 'flat_array' keys
                    with numpy array values where:
                    - flat_array:  numpy array containing concatenation
                                   of all nested arrays to be represented
                                   by this ragged array
                    - start_indices: unsiged integer numpy array the same
                                     length as the ragged array where values
                                     represent the index into flat_array where
                                     the corresponding ragged array element
                                     begins
            * RaggedArray: A RaggedArray instance to copy

        dtype: RaggedDtype or np.dtype or str or None (default None)
            Datatype to use to store underlying values from data.
            If none (the default) then dtype will be determined using the
            numpy.result_type function.
        copy : bool (default False)
            Whether to deep copy the input arrays. Only relevant when `data`
            has type `dict` or `RaggedArray`. When data is a `list` or
            `array`, input arrays are always copied.
        """
        if (isinstance(data, dict) and
                all(k in data for k in
                    ['start_indices', 'flat_array'])):

            _validate_ragged_properties(
                start_indices=data['start_indices'],
                flat_array=data['flat_array'])

            self._start_indices = data['start_indices']
            self._flat_array = data['flat_array']
            dtype = self._flat_array.dtype

            if copy:
                self._start_indices = self._start_indices.copy()
                self._flat_array = self._flat_array.copy()

        elif isinstance(data, RaggedArray):
            self._flat_array = data.flat_array
            self._start_indices = data.start_indices
            dtype = self._flat_array.dtype

            if copy:
                self._start_indices = self._start_indices.copy()
                self._flat_array = self._flat_array.copy()
        else:
            # Compute lengths
            index_len = len(data)
            buffer_len = sum(len(datum)
                             if not missing(datum)
                             else 0 for datum in data)

            # Compute necessary precision of start_indices array
            for nbits in [8, 16, 32, 64]:
                start_indices_dtype = 'uint' + str(nbits)
                max_supported = np.iinfo(start_indices_dtype).max
                if buffer_len <= max_supported:
                    break

            # infer dtype if not provided
            if dtype is None:
                non_missing = [np.atleast_1d(v)
                               for v in data if not missing(v)]
                if non_missing:
                    dtype = np.result_type(*non_missing)
                else:
                    dtype = 'float64'
            elif isinstance(dtype, RaggedDtype):
                dtype = dtype.subtype

            # Initialize representation arrays
            self._start_indices = np.zeros(index_len, dtype=start_indices_dtype)
            self._flat_array = np.zeros(buffer_len, dtype=dtype)

            # Populate arrays
            next_start_ind = 0
            for i, array_el in enumerate(data):
                # Compute element length
                n = len(array_el) if not missing(array_el) else 0

                # Update start indices
                self._start_indices[i] = next_start_ind

                # Do not assign when slice is empty avoiding possible
                # nan assignment to integer array
                if not n:
                    continue

                # Update flat array
                self._flat_array[next_start_ind:next_start_ind+n] = array_el

                # increment next start index
                next_start_ind += n

        self._dtype = RaggedDtype(dtype=dtype)

    def __eq__(self, other):
        if isinstance(other, RaggedArray):
            if len(other) != len(self):
                raise ValueError("""
Cannot check equality of RaggedArray values of unequal length
    len(ra1) == {len_ra1}
    len(ra2) == {len_ra2}""".format(
                    len_ra1=len(self),
                    len_ra2=len(other)))

            result = _eq_ragged_ragged(
                self.start_indices, self.flat_array,
                other.start_indices, other.flat_array)
        else:
            # Convert other to numpy array
            if not isinstance(other, np.ndarray):
                other_array = np.asarray(other)
            else:
                other_array = other

            if other_array.ndim == 1 and other_array.dtype.kind != 'O':

                # Treat as ragged scalar
                result = _eq_ragged_scalar(
                    self.start_indices, self.flat_array, other_array)
            elif (other_array.ndim == 1 and
                  other_array.dtype.kind == 'O' and
                  len(other_array) == len(self)):

                # Treat as vector
                result = _eq_ragged_ndarray1d(
                    self.start_indices, self.flat_array, other_array)
            elif (other_array.ndim == 2 and
                  other_array.dtype.kind != 'O' and
                  other_array.shape[0] == len(self)):

                # Treat rows as ragged elements
                result = _eq_ragged_ndarray2d(
                    self.start_indices, self.flat_array, other_array)
            else:
                raise ValueError("""
Cannot check equality of RaggedArray of length {ra_len} with:
    {other}""".format(ra_len=len(self), other=repr(other)))

        return result

    def __ne__(self, other):
        return np.logical_not(self == other)

    @property
    def flat_array(self):
        """
        numpy array containing concatenation of all nested arrays

        Returns
        -------
        np.ndarray
        """
        return self._flat_array

    @property
    def start_indices(self):
        """
        unsiged integer numpy array the same length as the ragged array where
        values represent the index into flat_array where the corresponding
        ragged array element begins

        Returns
        -------
        np.ndarray
        """
        return self._start_indices

    def __len__(self):
        return len(self._start_indices)

    def __getitem__(self, item):
        err_msg = ("Only integers, slices and integer or boolean"
                   "arrays are valid indices.")
        if isinstance(item, Integral):
            if item < -len(self) or item >= len(self):
                raise IndexError("{item} is out of bounds".format(item=item))
            else:
                # Convert negative item index
                if item < 0:
                    item += len(self)

                slice_start = self.start_indices[item]
                slice_end = (self.start_indices[item+1]
                             if item + 1 <= len(self) - 1
                             else len(self.flat_array))

                return (self.flat_array[slice_start:slice_end]
                        if slice_end!=slice_start
                        else np.nan)

        elif type(item) == slice:
            data = []
            selected_indices = np.arange(len(self))[item]

            for selected_index in selected_indices:
                data.append(self[selected_index])

            return RaggedArray(data, dtype=self.flat_array.dtype)

        elif isinstance(item, (np.ndarray, ExtensionArray, list, tuple)):
            if isinstance(item, (np.ndarray, ExtensionArray)):
                # Leave numpy and pandas arrays alone
                kind = item.dtype.kind
            else:
                # Convert others to pandas arrays
                item = pd.array(item)
                kind = item.dtype.kind

            if len(item) == 0:
                return self.take([], allow_fill=False)
            elif kind == 'b':
                # Check mask length is compatible
                if len(item) != len(self):
                    raise IndexError(
                        "Boolean index has wrong length: {} instead of {}"
                        .format(len(item), len(self))
                    )

                # check for NA values
                isna = pd.isna(item)
                if isna.any():
                    if Version(pd.__version__) > Version('1.0.1'):
                        item[isna] = False
                    else:
                        raise ValueError(
                            "Cannot mask with a boolean indexer containing NA values"
                        )

                data = []

                for i, m in enumerate(item):
                    if m:
                        data.append(self[i])

                return RaggedArray(data, dtype=self.flat_array.dtype)
            elif kind in ('i', 'u'):
                if any(pd.isna(item)):
                    raise ValueError(
                        "Cannot index with an integer indexer containing NA values"
                    )
                return self.take(item, allow_fill=False)
            else:
                raise IndexError(err_msg)
        else:
            raise IndexError(err_msg)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return RaggedArray(scalars, dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        return RaggedArray(
            [_RaggedElement.array_or_nan(v) for v in values],
            dtype=original.flat_array.dtype)

    def _as_ragged_element_array(self):
        return np.array([_RaggedElement.ragged_or_nan(self[i])
                         for i in range(len(self))])

    def _values_for_factorize(self):
        return self._as_ragged_element_array(), np.nan

    def _values_for_argsort(self):
        return self._as_ragged_element_array()

    def unique(self):
        from pandas import unique

        uniques = unique(self._as_ragged_element_array())
        return self._from_sequence(
            [_RaggedElement.array_or_nan(v) for v in uniques],
            dtype=self.dtype)

    def fillna(self, value=None, method=None, limit=None):
        # Override in RaggedArray to handle ndarray fill values
        from pandas.util._validators import validate_fillna_kwargs
        from pandas.core.missing import get_fill_func

        value, method = validate_fillna_kwargs(value, method)

        mask = self.isna()

        if isinstance(value, RaggedArray):
            if len(value) != len(self):
                raise ValueError("Length of 'value' does not match. Got ({}) "
                                 " expected {}".format(len(value), len(self)))
            value = value[mask]

        if mask.any():
            if method is not None:
                func = get_fill_func(method)
                new_values = func(self.astype(object), limit=limit,
                                  mask=mask)
                new_values = self._from_sequence(new_values, dtype=self.dtype)
            else:
                # fill with value
                new_values = list(self)
                mask_indices, = np.where(mask)
                for ind in mask_indices:
                    new_values[ind] = value

                new_values = self._from_sequence(new_values, dtype=self.dtype)
        else:
            new_values = self.copy()
        return new_values

    def shift(self, periods=1, fill_value=None):
        # Override in RaggedArray to handle ndarray fill values

        # Note: this implementation assumes that `self.dtype.na_value` can be
        # stored in an instance of your ExtensionArray with `self.dtype`.
        if not len(self) or periods == 0:
            return self.copy()

        if fill_value is None:
            fill_value = np.nan

        empty = self._from_sequence(
            [fill_value] * min(abs(periods), len(self)),
            dtype=self.dtype
        )
        if periods > 0:
            a = empty
            b = self[:-periods]
        else:
            a = self[abs(periods):]
            b = empty
        return self._concat_same_type([a, b])

    def searchsorted(self, value, side="left", sorter=None):
        arr = self._as_ragged_element_array()
        if isinstance(value, RaggedArray):
            search_value = value._as_ragged_element_array()
        else:
            search_value = _RaggedElement(value)
        return arr.searchsorted(search_value, side=side, sorter=sorter)

    def isna(self):
        stop_indices = np.hstack([self.start_indices[1:],
                                  [len(self.flat_array)]])

        element_lengths = stop_indices - self.start_indices
        return element_lengths == 0

    def take(self, indices, allow_fill=False, fill_value=None):
        if allow_fill:
            invalid_inds = [i for i in indices if i < -1]
            if invalid_inds:
                raise ValueError("""
Invalid indices for take with allow_fill True: {inds}""".format(
                    inds=invalid_inds[:9]))
            sequence = [self[i] if i >= 0 else fill_value
                        for i in indices]
        else:
            if len(self) == 0 and len(indices) > 0:
                raise IndexError(
                    "cannot do a non-empty take from an empty axis|out of bounds"
                )

            sequence = [self[i] for i in indices]

        return RaggedArray(sequence, dtype=self.flat_array.dtype)

    def copy(self, deep=False):
        data = dict(
            flat_array=self.flat_array,
            start_indices=self.start_indices)

        return RaggedArray(data, copy=deep)

    @classmethod
    def _concat_same_type(cls, to_concat):
        # concat flat_arrays
        flat_array = np.hstack([ra.flat_array for ra in to_concat])

        # offset and concat start_indices
        offsets = np.hstack([
            [0],
            np.cumsum([len(ra.flat_array) for ra in to_concat[:-1]])])

        start_indices = np.hstack([ra.start_indices + offset
                                   for offset, ra in zip(offsets, to_concat)])

        return RaggedArray(dict(
            flat_array=flat_array, start_indices=start_indices),
            copy=False)

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return (self._flat_array.nbytes +
                self._start_indices.nbytes)

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, RaggedDtype):
            if copy:
                return self.copy()
            return self

        elif is_extension_array_dtype(dtype):
            return dtype.construct_array_type()._from_sequence(
                np.asarray(self))

        return np.array([v for v in self], dtype=dtype, copy=copy)

    def tolist(self):
        # Based on pandas ExtensionArray.tolist
        if self.ndim > 1:
            return [item.tolist() for item in self]
        else:
            return list(self)

    def __array__(self, dtype=None):
        dtype = np.dtype(object) if dtype is None else np.dtype(dtype)
        return np.asarray(self.tolist(), dtype=dtype)


@jit(nopython=True, nogil=True)
def _eq_ragged_ragged(start_indices1,
                      flat_array1,
                      start_indices2,
                      flat_array2):
    """
    Compare elements of two ragged arrays of the same length

    Parameters
    ----------
    start_indices1: ndarray
        start indices of a RaggedArray 1
    flat_array1: ndarray
        flat_array property of a RaggedArray 1
    start_indices2: ndarray
        start indices of a RaggedArray 2
    flat_array2: ndarray
        flat_array property of a RaggedArray 2

    Returns
    -------
    mask: ndarray
        1D bool array of same length as inputs with elements True when
        corresponding elements are equal, False otherwise
    """
    n = len(start_indices1)
    m1 = len(flat_array1)
    m2 = len(flat_array2)

    result = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        # Extract inds for ra1
        start_index1 = start_indices1[i]
        stop_index1 = start_indices1[i + 1] if i < n - 1 else m1
        len_1 = stop_index1 - start_index1

        # Extract inds for ra2
        start_index2 = start_indices2[i]
        stop_index2 = start_indices2[i + 1] if i < n - 1 else m2
        len_2 = stop_index2 - start_index2

        if len_1 != len_2:
            el_equal = False
        else:
            el_equal = True
            for flat_index1, flat_index2 in \
                    zip(range(start_index1, stop_index1),
                        range(start_index2, stop_index2)):
                el_1 = flat_array1[flat_index1]
                el_2 = flat_array2[flat_index2]
                el_equal &= el_1 == el_2

        result[i] = el_equal

    return result


@jit(nopython=True, nogil=True)
def _eq_ragged_scalar(start_indices, flat_array, val):
    """
    Compare elements of a RaggedArray with a scalar array

    Parameters
    ----------
    start_indices: ndarray
        start indices of a RaggedArray
    flat_array: ndarray
        flat_array property of a RaggedArray
    val: ndarray

    Returns
    -------
    mask: ndarray
        1D bool array of same length as inputs with elements True when
        ragged element equals scalar val, False otherwise.
    """
    n = len(start_indices)
    m = len(flat_array)
    cols = len(val)
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        start_index = start_indices[i]
        stop_index = start_indices[i+1] if i < n - 1 else m

        if stop_index - start_index != cols:
            el_equal = False
        else:
            el_equal = True
            for val_index, flat_index in \
                    enumerate(range(start_index, stop_index)):
                el_equal &= flat_array[flat_index] == val[val_index]
        result[i] = el_equal

    return result


def _eq_ragged_ndarray1d(start_indices, flat_array, a):
    """
    Compare a RaggedArray with a 1D numpy object array of the same length

    Parameters
    ----------
    start_indices: ndarray
        start indices of a RaggedArray
    flat_array: ndarray
        flat_array property of a RaggedArray
    a: ndarray
        1D numpy array of same length as ra

    Returns
    -------
    mask: ndarray
        1D bool array of same length as input with elements True when
        corresponding elements are equal, False otherwise

    Notes
    -----
    This function is not numba accelerated because it, by design, inputs
    a numpy object array
    """

    n = len(start_indices)
    m = len(flat_array)
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        start_index = start_indices[i]
        stop_index = start_indices[i + 1] if i < n - 1 else m
        a_val = a[i]
        if (a_val is None or
                (np.isscalar(a_val) and np.isnan(a_val)) or
                len(a_val) == 0):
            result[i] = start_index == stop_index
        else:
            result[i] = np.array_equal(flat_array[start_index:stop_index],
                                       a_val)

    return result


@jit(nopython=True, nogil=True)
def _eq_ragged_ndarray2d(start_indices, flat_array, a):
    """
    Compare a RaggedArray with rows of a 2D numpy object array

    Parameters
    ----------
    start_indices: ndarray
        start indices of a RaggedArray
    flat_array: ndarray
        flat_array property of a RaggedArray
    a: ndarray
        A 2D numpy array where the length of the first dimension matches the
        length of the RaggedArray

    Returns
    -------
    mask: ndarray
        1D bool array of same length as input RaggedArray with elements True
        when corresponding elements of ra equal corresponding row of `a`
    """
    n = len(start_indices)
    m = len(flat_array)
    cols = a.shape[1]

    # np.bool is an alias for Python's built-in bool type, np.bool_ is the
    # numpy type that numba recognizes
    result = np.zeros(n, dtype=np.bool_)
    for row in range(n):
        start_index = start_indices[row]
        stop_index = start_indices[row + 1] if row < n - 1 else m

        # Check equality
        if stop_index - start_index != cols:
            el_equal = False
        else:
            el_equal = True
            for col, flat_index in enumerate(range(start_index, stop_index)):
                el_equal &= flat_array[flat_index] == a[row, col]
        result[row] = el_equal
    return result


@jit(nopython=True, nogil=True)
def _lexograph_lt(a1, a2):
    """
    Compare two 1D numpy arrays lexographically
    Parameters
    ----------
    a1: ndarray
        1D numpy array
    a2: ndarray
        1D numpy array

    Returns
    -------
    comparison:
        True if a1 < a2, False otherwise
    """
    for e1, e2 in zip(a1, a2):
        if e1 < e2:
            return True
        elif e1 > e2:
            return False
    return len(a1) < len(a2)


def ragged_array_non_empty(dtype):
    return RaggedArray([[1], [1, 2]], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(RaggedDtype)(ragged_array_non_empty)
