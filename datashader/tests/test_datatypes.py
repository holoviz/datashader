import pytest
import numpy as np
import pandas as pd
from datashader.datatypes import RaggedDtype, RaggedArray


# Testing helpers
# ---------------
def assert_ragged_arrays_equal(ra1, ra2):
    assert np.array_equal(ra1.mask, ra2.mask)
    assert np.array_equal(ra1.start_indices, ra2.start_indices)
    assert np.array_equal(ra1.flat_array, ra2.flat_array)
    assert np.array_equal(ra1.flat_array.dtype, ra2.flat_array.dtype)


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

    # Check mask
    assert rarray.mask.dtype == 'bool'
    assert np.array_equal(
        rarray.mask,
        np.array([False, False, False, True, False], dtype='bool'))

    # Check len
    assert len(rarray) == 5

    # Check isna
    assert rarray.isna().dtype == 'bool'
    assert np.array_equal(
        rarray.isna(), [False, False, False, True, False])

    # Check nbytes
    expected = (
            9 * np.int32().nbytes +  # flat_array
            5 * np.uint8().nbytes +  # start_indices
            5                        # mask
    )
    assert rarray.nbytes == expected

    # Check dtype
    assert rarray.dtype == RaggedDtype


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


# __getitem__
# -----------
def test_get_item_scalar():
    arg = [[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]]
    rarray = RaggedArray(arg, dtype='float16')

    # Forward
    for i, expected in enumerate(arg):
        result = rarray[i]
        if expected is None:
            assert result is None
        else:
            assert result.dtype == 'float16'
            assert np.array_equal(result, expected)

    # Reversed
    for i, expected in enumerate(arg):
        result = rarray[i - 5]
        if expected is None:
            assert result is None
        else:
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


# _from_factorized
# ----------------
def test_factorization():
    arg = np.array([[1, 2], [], [1, 2], None, [11, 22, 33, 44]])
    rarray = RaggedArray(arg, dtype='int16')
    labels, uniques = rarray.factorize()

    assert np.array_equal(labels, [0, 1, 0, -1, 2])
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
