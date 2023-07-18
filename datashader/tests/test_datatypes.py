from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version

from datashader.datatypes import RaggedDtype, RaggedArray

# Import pandas fixtures so that overridden tests have access to them
from pandas.tests.extension.conftest import *  # noqa (fixture import)

# Testing helpers
# ---------------
def assert_ragged_arrays_equal(ra1, ra2):
    assert np.array_equal(ra1.start_indices, ra2.start_indices)
    assert np.array_equal(ra1.flat_array, ra2.flat_array)
    assert ra1.flat_array.dtype == ra2.flat_array.dtype

    # Make sure ragged elements are equal when iterated over
    for a1, a2 in zip(ra1, ra2):
        np.testing.assert_array_equal(a1, a2)


# Test constructor and properties
# -------------------------------
def test_construct_ragged_dtype():
    dtype = RaggedDtype()
    assert dtype.type == np.ndarray
    assert dtype.name == 'Ragged[{subtype}]'.format(subtype=dtype.subtype)
    assert dtype.kind == 'O'


def test_construct_ragged_array():
    rarray = RaggedArray([[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]],
                         dtype='int32')

    # Check flat array
    assert rarray.flat_array.dtype == 'int32'
    assert np.array_equal(
        rarray.flat_array,
        np.array([1, 2, 10, 20, 30, 11, 22, 33, 44], dtype='int32'))

    # Check start indices
    assert rarray.start_indices.dtype == 'uint8'
    assert np.array_equal(
        rarray.start_indices,
        np.array([0, 2, 2, 5, 5], dtype='uint64'))

    # Check len
    assert len(rarray) == 5

    # Check isna
    assert rarray.isna().dtype == 'bool'
    assert np.array_equal(
        rarray.isna(), [False, True, False, True, False])

    # Check nbytes
    expected = (
            9 * np.int32().nbytes +  # flat_array
            5 * np.uint8().nbytes    # start_indices
    )
    assert rarray.nbytes == expected

    # Check dtype
    assert type(rarray.dtype) == RaggedDtype


def test_construct_ragged_array_from_ragged_array():
    rarray = RaggedArray([[1, 2], [], [10, 20, 30], np.nan, [11, 22, 33, 44]],
                         dtype='int32')

    result = RaggedArray(rarray)
    assert_ragged_arrays_equal(result, rarray)


def test_construct_ragged_array_fastpath():

    start_indices = np.array([0, 2, 5, 6, 6, 11], dtype='uint16')
    flat_array = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='float32')

    rarray = RaggedArray(
        dict(start_indices=start_indices, flat_array=flat_array))

    # Check that arrays were accepted unchanged
    assert np.array_equal(rarray.start_indices, start_indices)
    assert np.array_equal(rarray.flat_array, flat_array)

    # Check interpretation as ragged array
    object_array = np.asarray(rarray, dtype=object)
    expected_lists = [[0, 1], [2, 3, 4], [5], [], [6, 7, 8, 9, 10], []]
    expected_array = np.array([np.array(v, dtype='float32')
                               for v in expected_lists], dtype='object')

    assert len(object_array) == len(expected_array)
    for a1, a2 in zip(object_array, expected_array):
        np.testing.assert_array_equal(a1, a2)


def test_validate_ragged_array_fastpath():
    start_indices = np.array([0, 2, 5, 6, 6, 11], dtype='uint16')
    flat_array = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='float32')

    valid_dict = dict(start_indices=start_indices, flat_array=flat_array)

    # Valid args
    RaggedArray(valid_dict)

    # ## start_indices validation ##
    #
    # not ndarray
    with pytest.raises(ValueError) as ve:
        RaggedArray(dict(valid_dict, start_indices=25))
    ve.match('start_indices property of a RaggedArray')

    # not unsiged int
    with pytest.raises(ValueError) as ve:
        RaggedArray(dict(valid_dict,
                         start_indices=start_indices.astype('float32')))
    ve.match('start_indices property of a RaggedArray')

    # not 1d
    with pytest.raises(ValueError) as ve:
        RaggedArray(dict(valid_dict, start_indices=np.array([start_indices])))
    ve.match('start_indices property of a RaggedArray')

    # ## flat_array validation ##
    #
    # not ndarray
    with pytest.raises(ValueError) as ve:
        RaggedArray(dict(valid_dict, flat_array='foo'))
    ve.match('flat_array property of a RaggedArray')

    # not 1d
    with pytest.raises(ValueError) as ve:
        RaggedArray(dict(valid_dict, flat_array=np.array([flat_array])))
    ve.match('flat_array property of a RaggedArray')

    # ## start_indices out of bounds validation ##
    #
    bad_start_indices = start_indices.copy()
    bad_start_indices[-1] = 99
    with pytest.raises(ValueError) as ve:
        RaggedArray(dict(valid_dict, start_indices=bad_start_indices))
    ve.match('start_indices must be less than')


def test_start_indices_dtype():
    # The start_indices dtype should be an unsiged int that is only as large
    # as needed to handle the length of the flat array

    # Empty
    rarray = RaggedArray([[]], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint8')
    np.testing.assert_array_equal(rarray.start_indices, [0])

    # Small
    rarray = RaggedArray([[23, 24]], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint8')
    np.testing.assert_array_equal(rarray.start_indices, [0])

    # Max uint8
    max_uint8 = np.iinfo('uint8').max
    rarray = RaggedArray([np.zeros(max_uint8), []], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint8')
    np.testing.assert_array_equal(rarray.start_indices, [0, max_uint8])

    # Min uint16
    rarray = RaggedArray([np.zeros(max_uint8 + 1), []], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint16')
    np.testing.assert_array_equal(rarray.start_indices, [0, max_uint8 + 1])

    # Max uint16
    max_uint16 = np.iinfo('uint16').max
    rarray = RaggedArray([np.zeros(max_uint16), []], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint16')
    np.testing.assert_array_equal(rarray.start_indices, [0, max_uint16])

    # Min uint32
    rarray = RaggedArray([np.zeros(max_uint16 + 1), []], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint32')
    np.testing.assert_array_equal(rarray.start_indices, [0, max_uint16 + 1])


@pytest.mark.parametrize('arg,expected', [
    ([np.array([1, 2], dtype='int64')], 'int64'),
    ([[True], [False, True]], 'bool'),
    (np.array([np.array([1, 2], dtype='int8'),
               np.array([1, 2], dtype='int32')]), 'int32'),
    ([[3.2], [2]], 'float64'),
    ([np.array([3.2], dtype='float16'),
      np.array([2], dtype='float32')], 'float32')
])
def test_flat_array_type_inference(arg, expected):
    rarray = RaggedArray(arg)
    assert rarray.flat_array.dtype == np.dtype(expected)


# isna
# -----
def test_isna():
    rarray = RaggedArray([[], [1, 3], [10, 20, 30],
                          None, [11, 22, 33, 44], []], dtype='int32')

    np.testing.assert_array_equal(rarray.isna(),
                          np.array([True, False, False, True, False, True]))


# __getitem__
# -----------
def test_get_item_scalar():
    arg = [[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]]
    rarray = RaggedArray(arg, dtype='float16')

    # Forward
    for i, expected in enumerate(arg):
        result = rarray[i]
        if expected is None:
            expected = np.array([], dtype='float16')

        if isinstance(result, np.ndarray):
            assert result.dtype == 'float16'
        else:
            assert np.isnan(result)

        np.testing.assert_array_equal(result, expected)

    # Reversed
    for i, expected in enumerate(arg):
        result = rarray[i - 5]
        if expected is None:
            expected = np.array([], dtype='float16')

        if isinstance(result, np.ndarray):
            assert result.dtype == 'float16'
        else:
            assert np.isnan(result)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('index', [-1000, -6, 5, 1000])
def test_get_item_scalar_out_of_bounds(index):
    rarray = RaggedArray([[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]])
    with pytest.raises(IndexError):
        rarray[index]


def test_get_item_slice():
    arg = [[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]]
    rarray = RaggedArray(arg, dtype='int16')

    # Slice everything
    assert_ragged_arrays_equal(rarray[:], rarray)

    # Slice all but the first
    assert_ragged_arrays_equal(
        rarray[1:], RaggedArray(arg[1:], dtype='int16'))

    # Slice all but the last
    assert_ragged_arrays_equal(
        rarray[:-1], RaggedArray(arg[:-1], dtype='int16'))

    # Slice middle
    assert_ragged_arrays_equal(
        rarray[2:-1], RaggedArray(arg[2:-1], dtype='int16'))

    # Empty slice
    assert_ragged_arrays_equal(
        rarray[2:1], RaggedArray(arg[2:1], dtype='int16'))


@pytest.mark.parametrize('mask', [
    [1, 1, 1, 1, 1],
    [0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0]
])
def test_get_item_mask(mask):
    arg = np.array([[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]], dtype=object)
    rarray = RaggedArray(arg, dtype='int16')
    mask = np.array(mask, dtype='bool')

    assert_ragged_arrays_equal(
        rarray[mask],
        RaggedArray(arg[mask], dtype='int16'))


@pytest.mark.parametrize('inds', [
    [1, 2, 1, 4],
    np.array([1, 2, 1, 4]),
    [],
    np.array([], dtype='int32'),
    [4, 3, 2, 1, 0]
])
def test_get_item_list(inds):
    arg = np.array([[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]], dtype=object)
    rarray = RaggedArray(arg, dtype='int16')

    assert_ragged_arrays_equal(
        rarray[inds],
        RaggedArray(arg[inds], dtype='int16'))


# _from_factorized
# ----------------
def test_factorization():
    arg = np.array([[1, 2], [], [1, 2], None, [11, 22, 33, 44]], dtype=object)
    rarray = RaggedArray(arg, dtype='int16')
    labels, uniques = rarray.factorize()

    np.testing.assert_array_equal(labels, [0, -1, 0, -1, 1])
    assert_ragged_arrays_equal(
        uniques, RaggedArray([[1, 2], [11, 22, 33, 44]], dtype='int16'))


# _from_sequence
# --------------
def test_from_sequence():
    sequence = [[1, 2], [], [1, 2], None, [11, 22, 33, 44]]
    rarray = RaggedArray._from_sequence(sequence)

    assert_ragged_arrays_equal(
        rarray, RaggedArray(sequence))


# copy
# ----
def test_copy():
    # Create reference ragged array
    original = RaggedArray._from_sequence(
        [[1, 2], [], [1, 2], None, [11, 22, 33, 44]])

    # Copy reference array
    copied = original.copy(deep=True)

    # Make sure arrays are equal
    assert_ragged_arrays_equal(original, copied)

    # Modify buffer in original
    original.flat_array[0] = 99
    assert original.flat_array[0] == 99

    # Make sure copy was not modified
    assert copied.flat_array[0] == 1


# take
# ----
def test_take():
    #
    rarray = RaggedArray._from_sequence(
        [[1, 2], [], [10, 20], None, [11, 22, 33, 44]])

    # allow_fill False
    result = rarray.take([0, 2, 1, -1, -2, 0], allow_fill=False)
    expected = RaggedArray(
        [[1, 2], [10, 20], [], [11, 22, 33, 44], None, [1, 2]])
    assert_ragged_arrays_equal(result, expected)

    # allow fill True
    result = rarray.take([0, 2, 1, -1, -1, 0], allow_fill=True)
    expected = RaggedArray(
        [[1, 2], [10, 20], [], None, None, [1, 2]])
    assert_ragged_arrays_equal(result, expected)


# _concat_same_type
# -----------------
def test_concat_same_type():
    arg1 = [[1, 2], [], [10, 20], None, [11, 22, 33, 44]]
    rarray1 = RaggedArray(arg1, dtype='float32')

    arg2 = [[100, 200], None, [99, 100, 101]]
    rarray2 = RaggedArray(arg2, dtype='float32')

    arg3 = [None, [27, 28]]
    rarray3 = RaggedArray(arg3, dtype='float32')

    result = RaggedArray._concat_same_type([rarray1, rarray2, rarray3])
    expected = RaggedArray(arg1 + arg2 + arg3, dtype='float32')

    assert_ragged_arrays_equal(result, expected)


# Test pandas operations
# ----------------------
def test_pandas_array_construction():
    arg = [[0, 1], [1, 2, 3, 4], None, [-1, -2]] * 2
    ra = pd.array(arg, dtype='ragged[int64]')

    expected = RaggedArray(arg, dtype='int64')
    assert_ragged_arrays_equal(ra, expected)


def test_series_construction():
    arg = [[0, 1], [1.0, 2, 3.0, 4], None, [-1, -2]] * 2
    rs = pd.Series(arg, dtype='Ragged[int64]')
    ra = rs.array

    expected = RaggedArray(arg, dtype='int64')
    assert_ragged_arrays_equal(ra, expected)


def test_concat_series():
    arg1 = [[1, 2], [], [10, 20], None, [11, 22, 33, 44]]
    s1 = pd.Series(arg1, dtype='ragged[int16]')

    arg2 = [[100, 200], None, [99, 100, 101]]
    s2 = pd.Series(arg2, dtype='ragged[int16]')

    arg3 = [None, [27, 28]]
    s3 = pd.Series(arg3, dtype='ragged[int16]')

    s_concat = pd.concat([s1, s2, s3])

    expected = pd.Series(arg1+arg2+arg3,
                         dtype='ragged[int16]',
                         index=[0, 1, 2, 3, 4, 0, 1, 2, 0, 1])

    pd.testing.assert_series_equal(s_concat, expected)


# Array equality
# --------------
@pytest.mark.parametrize('scalar', [
    np.array([1, 2]), [1, 2]
])
def test_array_eq_scalar(scalar):
    # Build RaggedArray
    arg1 = [[1, 2], [], [1, 2], [1, 3], [11, 22, 33, 44]]
    ra = RaggedArray(arg1, dtype='int32')

    # Check equality
    result = ra == scalar
    expected = np.array([1, 0, 1, 0, 0], dtype='bool')
    np.testing.assert_array_equal(result, expected)

    # Check non-equality
    result_negated = ra != scalar
    expected_negated = ~expected
    np.testing.assert_array_equal(result_negated, expected_negated)


def test_array_eq_numpy1():
    # Build RaggedArray
    arg1 = [[1, 2], [], [1, 2], None, [11, 22, 33, 44]]

    # Construct arrays
    ra = RaggedArray(arg1, dtype='int32')
    npa = np.array([[1, 2], [2], [1, 2], None, [10, 20, 30, 40]],
                   dtype='object')

    # Check equality
    result = ra == npa
    expected = np.array([1, 0, 1, 1, 0], dtype='bool')
    np.testing.assert_array_equal(result, expected)

    # Check non-equality
    result_negated = ra != npa
    expected_negated = ~expected
    np.testing.assert_array_equal(result_negated, expected_negated)


def test_array_eq_numpy2d():
    # Construct arrays
    ra = RaggedArray([[1, 2], [1], [1, 2], None, [33, 44]],
                     dtype='int32')
    npa = np.array([[1, 2], [2, 3], [1, 2], [0, 1], [11, 22]],
                   dtype='int32')

    # Check equality
    result = ra == npa
    expected = np.array([1, 0, 1, 0, 0], dtype='bool')
    np.testing.assert_array_equal(result, expected)

    # Check non-equality
    result_negated = ra != npa
    expected_negated = ~expected
    np.testing.assert_array_equal(result_negated, expected_negated)


def test_array_eq_ragged():
    # Build RaggedArray
    arg1 = [[1, 2], [], [1, 2], [3, 2, 1], [11, 22, 33, 44]]
    ra1 = RaggedArray(arg1, dtype='int32')

    # Build RaggedArray
    arg2 = [[1, 2], [2, 3, 4, 5], [1, 2], [11, 22, 33], [11]]
    ra2 = RaggedArray(arg2, dtype='int32')

    # Check equality
    result = ra1 == ra2
    expected = np.array([1, 0, 1, 0, 0], dtype='bool')
    np.testing.assert_array_equal(result, expected)

    # Check non-equality
    result_negated = ra1 != ra2
    expected_negated = ~expected
    np.testing.assert_array_equal(result_negated, expected_negated)


@pytest.mark.parametrize('other', [
    'a string',  # Incompatible scalars
    32,
    RaggedArray([[0, 1], [2, 3, 4]]),  # RaggedArray of wrong length
    np.array([[0, 1], [2, 3, 4]], dtype='object'),  # 1D array wrong length
    np.array([[0, 1], [2, 3]], dtype='int32'),  # 2D array wrong row count
])
def test_equality_validation(other):
    # Build RaggedArray
    arg1 = [[1, 2], [], [1, 2], None, [11, 22, 33, 44]]
    ra1 = RaggedArray(arg1, dtype='int32')

    # invalid scalar
    with pytest.raises(ValueError, match="Cannot check equality"):
        ra1 == other


# Pandas-provided extension array tests
# -------------------------------------
# See https://pandas.pydata.org/docs/development/extending.html
@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return RaggedDtype()


@pytest.fixture
def data():
    """Length-100 array for this type.
        * data[0] and data[1] should both be non missing
        * data[0] and data[1] should not gbe equal
        """
    return RaggedArray(
        [[0, 1], [1, 2, 3, 4], [], [-1, -2], []]*20, dtype='float64')


@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.
    Parameters
    ----------
    data : fixture implementing `data`
    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """
    def gen(count):
        for _ in range(count):
            yield data
    return gen


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    return request.param


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    return RaggedArray([[], [-1, 0, 1]], dtype='int16')


@pytest.fixture(params=['data', 'data_missing'])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == 'data':
        return data
    elif request.param == 'data_missing':
        return data_missing


@pytest.fixture
def data_for_sorting():
    """Length-3 array with a known sort order.
    This should be three items [B, C, A] with
    A < B < C
    """
    return RaggedArray([[1, 0], [2, 0], [0, 0]])


@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known sort order.
    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return RaggedArray([[1, 0], [], [0, 0]])


@pytest.fixture
def data_for_grouping():
    """Data for factorization, grouping, and unique tests.
    Expected to be like [B, B, NA, NA, A, A, B, C]
    Where A < B < C and NA is missing
    """
    return RaggedArray(
        [[1, 0], [1, 0], [], [], [0, 0], [0, 0], [1, 0], [2, 0]])


@pytest.fixture
def na_cmp():
    return lambda x, y: (np.isscalar(x) and np.isnan(x) and
                         np.isscalar(y) and np.isnan(y))


@pytest.fixture
def na_value():
    return np.nan


# Subclass BaseDtypeTests to run pandas-provided extension array test suite
class TestRaggedConstructors(eb.BaseConstructorsTests):

    @pytest.mark.skip(reason="Constructing DataFrame from RaggedArray not supported")
    def test_from_dtype(self, data):
        pass

    @pytest.mark.skip(reason="passing scalar with index not supported")
    def test_series_constructor_scalar_with_index(self, data, dtype):
        pass


class TestRaggedDtype(eb.BaseDtypeTests):
    pass


class TestRaggedGetitem(eb.BaseGetitemTests):

    # Override testing methods that assume extension array scalars are
    # comparable using `==`. Replace with assert_array_equal.
    #
    # If pandas introduces a way to customize element equality tests
    # these overrides should be removed.
    def test_get(self, data):
        # GH 20882
        s = pd.Series(data, index=[2 * i for i in range(len(data))])
        np.testing.assert_array_equal(s.get(4), s.iloc[2])

        result = s.get([4, 6])
        expected = s.iloc[[2, 3]]
        self.assert_series_equal(result, expected)

        result = s.get(slice(2))
        expected = s.iloc[[0, 1]]
        self.assert_series_equal(result, expected)

        assert s.get(-1) is None
        assert s.get(s.index.max() + 1) is None

        s = pd.Series(data[:6], index=list('abcdef'))
        np.testing.assert_array_equal(s.get('c'), s.iloc[2])

        result = s.get(slice('b', 'd'))
        expected = s.iloc[[1, 2, 3]]
        self.assert_series_equal(result, expected)

        result = s.get('Z')
        assert result is None

        np.testing.assert_array_equal(s.get(4), s.iloc[4])
        np.testing.assert_array_equal(s.get(-1), s.iloc[-1])
        assert s.get(len(s)) is None

    def test_take_sequence(self, data):
        result = pd.Series(data)[[0, 1, 3]]
        np.testing.assert_array_equal(result.iloc[0], data[0])
        np.testing.assert_array_equal(result.iloc[1], data[1])
        np.testing.assert_array_equal(result.iloc[2], data[3])

    def test_take(self, data, na_value, na_cmp):
        result = data.take([0, -1])
        np.testing.assert_array_equal(result.dtype, data.dtype)
        np.testing.assert_array_equal(result[0], data[0])
        np.testing.assert_array_equal(result[1], data[-1])

        result = data.take([0, -1], allow_fill=True, fill_value=na_value)
        np.testing.assert_array_equal(result[0], data[0])
        assert na_cmp(result[1], na_value)

        with pytest.raises(IndexError, match="out of bounds"):
            data.take([len(data) + 1])

    def test_item(self, data):
        # https://github.com/pandas-dev/pandas/pull/30175
        s = pd.Series(data)
        result = s[:1].item()
        np.testing.assert_array_equal(result, data[0])

        msg = "can only convert an array of size 1 to a Python scalar"
        with pytest.raises(ValueError, match=msg):
            s[:0].item()

        with pytest.raises(ValueError, match=msg):
            s.item()

    @pytest.mark.skip(
        reason="Ellipsis not supported in RaggedArray.__getitem__"
    )
    def test_getitem_ellipsis_and_slice(self, data):
        pass

    # Ellipsis, numpy.newaxis, None, not supported in RaggedArray.__getitem__
    # so the error message raised RaggedArray.__getitem__ isn't the same as
    # the one raised by the base extension.
    @pytest.mark.skip(reason="RaggedArray.__getitem__ raises a different error message")
    def test_getitem_invalid(self, data):
        pass

    @pytest.mark.skip(reason="Can't autoconvert ragged array to numpy array")
    def test_getitem_series_integer_with_missing_raises(self, data, idx):
        pass


class TestRaggedGroupby(eb.BaseGroupbyTests):
    @pytest.mark.skip(reason="agg not supported")
    def test_groupby_agg_extension(self):
        pass

    @pytest.mark.skip(reason="numpy.ndarray unhashable")
    def test_groupby_extension_transform(self):
        pass

    @pytest.mark.skip(reason="agg not supported")
    def test_groupby_extension_agg(self):
        pass

    @pytest.mark.skip(
        reason="numpy.ndarray unhashable and buffer wrong number of dims")
    def test_groupby_extension_apply(self):
        pass


class TestRaggedInterface(eb.BaseInterfaceTests):
    # Add array equality
    def test_array_interface(self, data):
        result = np.array(data, dtype=object)
        np.testing.assert_array_equal(result[0], data[0])

        result = np.array(data, dtype=object)
        expected = np.array(list(data), dtype=object)

        for a1, a2 in zip(result, expected):
            if np.isscalar(a1):
                assert np.isnan(a1) and np.isnan(a2)
            else:
                np.testing.assert_array_equal(a1, a2)

    # # NotImplementedError: 'RaggedArray' does not support __setitem__
    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_copy(self):
        pass

    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_view(self):
        pass

    @pytest.mark.skipif(Version(pd.__version__) < Version("1.4"), reason="Added in pandas 1.4")
    def test_tolist(self, data):
        result = data.tolist()
        expected = list(data)
        assert isinstance(result, list)
        for r, e in zip(result, expected):
            assert np.array_equal(r, e, equal_nan=True)


class TestRaggedMethods(eb.BaseMethodsTests):

    # # AttributeError: 'RaggedArray' object has no attribute 'value_counts'
    @pytest.mark.skip(reason="value_counts not supported")
    def test_value_counts(self):
        pass

    @pytest.mark.skip(reason="value_counts not supported")
    def test_value_counts_with_normalize(self):
        pass

    @pytest.mark.skip(reason="shift not supported")
    def test_shift_0_periods(self):
        pass

    # Add array equality
    @pytest.mark.parametrize('box', [pd.Series, lambda x: x])
    @pytest.mark.parametrize('method', [lambda x: x.unique(), pd.unique])
    def test_unique(self, data, box, method):
        duplicated = box(data._from_sequence([data[0], data[0]]))

        result = method(duplicated)

        assert len(result) == 1
        assert isinstance(result, type(data))
        np.testing.assert_array_equal(result[0], duplicated[0])

    # Pandas raises
    #   ValueError: invalid fill value with a <class 'numpy.ndarray'>
    @pytest.mark.skip(reason="pandas cannot fill with ndarray")
    def test_fillna_copy_frame(self):
        pass

    @pytest.mark.skip(reason="pandas cannot fill with ndarray")
    def test_fillna_copy_series(self):
        pass

    # Ragged array elements don't support binary operators
    @pytest.mark.skip(reason="ragged does not support <= on elements")
    def test_combine_le(self):
        pass

    @pytest.mark.skip(reason="ragged does not support + on elements")
    def test_combine_add(self):
        pass

    # Block manager error:
    #   ValueError: setting an array element with a sequence.
    @pytest.mark.skip(reason="combine_first not supported")
    def test_combine_first(self):
        pass

    @pytest.mark.skip(
        reason="Searchsorted seems not implemented for custom extension arrays"
    )
    def test_searchsorted(self):
        pass

    @pytest.mark.skip(reason="ragged cannot be used as categorical")
    def test_sort_values_frame(self):
        pass

    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_where_series(self):
        pass

class TestRaggedPrinting(eb.BasePrintingTests):
    @pytest.mark.skip(reason="Can't autoconvert ragged array to numpy array")
    def test_dataframe_repr(self):
        pass

    @pytest.mark.skip(reason="Can't autoconvert ragged array to numpy array")
    def test_series_repr(self):
        pass


class TestRaggedMissing(eb.BaseMissingTests):
    # Pandas doesn't like using an ndarray as fill value.
    # Errors like:
    #   ValueError: invalid fill value with a <class 'numpy.ndarray'>
    @pytest.mark.skip(reason="Can't fill with ndarray")
    def test_fillna_series(self):
        pass

    @pytest.mark.skip(reason="Can't fill with ndarray")
    def test_fillna_frame(self):
        pass

    @pytest.mark.skip(reason="Can't fill with nested sequences")
    def test_fillna_limit_pad(self):
        pass

    @pytest.mark.skip(reason="Can't fill with nested sequences")
    def test_fillna_limit_backfill(self):
        pass

    @pytest.mark.skip(reason="Can't fill with nested sequences")
    def test_fillna_series_method(self):
        pass


class TestRaggedReshaping(eb.BaseReshapingTests):
    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_ravel(self):
        pass

    @pytest.mark.skip(reason="transpose with numpy array elements seems not supported")
    def test_transpose(self):
        pass

    @pytest.mark.skip(reason="transpose with numpy array elements seems not supported")
    def test_transpose_frame(self):
        pass

