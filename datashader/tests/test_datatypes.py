import pytest
import numpy as np
import pandas as pd
from pandas.tests.extension.base import BaseDtypeTests

from datashader.datatypes import RaggedDtype, RaggedArray


# Testing helpers
# ---------------
def assert_ragged_arrays_equal(ra1, ra2):
    assert np.array_equal(ra1.start_indices, ra2.start_indices)
    assert np.array_equal(ra1.flat_array, ra2.flat_array)
    assert np.array_equal(ra1.flat_array.dtype, ra2.flat_array.dtype)

    # Make sure ragged elements are equal when iterated over
    for a1, a2 in zip(ra1, ra2):
        assert np.array_equal(a1, a2)


# Test constructor and properties
# -------------------------------
def test_construct_ragged_dtype():
    dtype = RaggedDtype()
    assert dtype.type == np.ndarray
    assert dtype.name == 'ragged'
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
    rarray = RaggedArray([[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]],
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
    object_array = np.asarray(rarray)
    expected_lists = [[0, 1], [2, 3, 4], [5], [], [6, 7, 8, 9, 10], []]
    expected_array = np.array([np.array(v, dtype='float32')
                               for v in expected_lists], dtype='object')

    assert len(object_array) == len(expected_array)
    for a1, a2 in zip(object_array, expected_array):
        assert np.array_equal(a1, a2)


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
    assert np.array_equal(rarray.start_indices, [0])

    # Small
    rarray = RaggedArray([[23, 24]], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint8')
    assert np.array_equal(rarray.start_indices, [0])

    # Max uint8
    max_uint8 = np.iinfo('uint8').max
    rarray = RaggedArray([np.zeros(max_uint8), []], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint8')
    assert np.array_equal(rarray.start_indices, [0, max_uint8])

    # Min uint16
    rarray = RaggedArray([np.zeros(max_uint8 + 1), []], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint16')
    assert np.array_equal(rarray.start_indices, [0, max_uint8 + 1])

    # Max uint16
    max_uint16 = np.iinfo('uint16').max
    rarray = RaggedArray([np.zeros(max_uint16), []], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint16')
    assert np.array_equal(rarray.start_indices, [0, max_uint16])

    # Min uint32
    rarray = RaggedArray([np.zeros(max_uint16 + 1), []], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint32')
    assert np.array_equal(rarray.start_indices, [0, max_uint16 + 1])


@pytest.mark.parametrize('arg,expected', [
    ([[1, 2]], 'int64'),
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

    assert np.array_equal(rarray.isna(),
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

        assert result.dtype == 'float16'
        assert np.array_equal(result, expected)

    # Reversed
    for i, expected in enumerate(arg):
        result = rarray[i - 5]
        if expected is None:
            expected = np.array([], dtype='float16')

        assert result.dtype == 'float16'
        assert np.array_equal(result, expected)


@pytest.mark.parametrize('index', [-1000, -6, 5, 1000])
def test_get_item_scalar_out_of_bounds(index):
    rarray = RaggedArray([[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]])
    with pytest.raises(IndexError) as e:
        result = rarray[index]


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
    arg = np.array([[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]])
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
    arg = np.array([[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]])
    rarray = RaggedArray(arg, dtype='int16')

    assert_ragged_arrays_equal(
        rarray[inds],
        RaggedArray(arg[inds], dtype='int16'))


# _from_factorized
# ----------------
def test_factorization():
    arg = np.array([[1, 2], [], [1, 2], None, [11, 22, 33, 44]])
    rarray = RaggedArray(arg, dtype='int16')
    labels, uniques = rarray.factorize()

    assert np.array_equal(labels, [0, 1, 0, 1, 2])
    assert_ragged_arrays_equal(
        uniques, RaggedArray([[1, 2], [], [11, 22, 33, 44]], dtype='int16'))


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
    ra = pd.array(arg, dtype='ragged')

    expected = RaggedArray(arg)
    assert_ragged_arrays_equal(ra, expected)


def test_series_construction():
    arg = [[0, 1], [1, 2, 3, 4], None, [-1, -2]] * 2
    rs = pd.Series(arg, dtype='ragged')
    ra = rs.array

    expected = RaggedArray(arg)
    assert_ragged_arrays_equal(ra, expected)


def test_concat_series():
    arg1 = [[1, 2], [], [10, 20], None, [11, 22, 33, 44]]
    s1 = pd.Series(arg1, dtype='ragged')

    arg2 = [[100, 200], None, [99, 100, 101]]
    s2 = pd.Series(arg2, dtype='ragged')

    arg3 = [None, [27, 28]]
    s3 = pd.Series(arg3, dtype='ragged')

    s_concat = pd.concat([s1, s2, s3])

    expected = pd.Series(arg1+arg2+arg3,
                         dtype='ragged',
                         index=[0, 1, 2, 3, 4, 0, 1, 2, 0, 1])

    pd.testing.assert_series_equal(s_concat, expected)


# Pandas-provided extension array tests
# -------------------------------------
# See http://pandas-docs.github.io/pandas-docs-travis/extending.html
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
        [[0, 1], [1, 2, 3, 4], [], None, [-1, -2]]*20, dtype='float64')


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    return RaggedArray([None, [-1, 0, 1]], dtype='int16')


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
    return RaggedArray([[1, 0], None, [0, 0]])


@pytest.fixture
def data_for_grouping():
    """Data for factorization, grouping, and unique tests.
    Expected to be like [B, B, NA, NA, A, A, B, C]
    Where A < B < C and NA is missing
    """
    return RaggedArray(
        [[1, 0], [1, 0], None, None, [0, 0], [0, 0], [1, 0], [2, 0]])


# Subclass BaseDtypeTests to run pandas-provided extension array test suite
class TestRaggedDtype(BaseDtypeTests):
    pass
