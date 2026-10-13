from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray

from datashader.datashape import dshape
from datashader.utils import (
    Dispatcher, _extract_values_and_mask, apply, calc_res, dshape_from_pandas, isreal,
    orient_array,
)


def test_Dispatcher():
    foo = Dispatcher()
    foo.register(int, lambda a, b, c=1: a + b + c)
    foo.register(float, lambda a, b, c=1: a - b + c)
    foo.register(object, lambda a, b, c=1: 10)

    class Bar:
        pass
    b = Bar()
    assert foo(1, 2) == 4
    assert foo(1, 2.0, 3.0) == 6.0
    assert foo(1.0, 2.0, 3.0) == 2.0
    assert foo(b, 2) == 10


def test_isreal():
    assert isreal('int32')
    assert isreal(dshape('int32'))
    assert isreal('?int32')
    assert isreal('float64')
    assert not isreal('complex64')
    assert not isreal('{x: int64, y: float64}')


def test_apply():
    def f(a, b, c=1, d=2):
        return a + b + c + d
    assert apply(f, (1, 2,)) == 6
    assert apply(f, (1, 2,), dict(c=3)) == 8


def test_calc_res():
    x = [5, 7]
    y = [0, 1]
    z = [[0, 0], [0, 0]]
    dims = ('y', 'x')

    # x and y increasing
    xarr = DataArray(z, coords=dict(x=x, y=y), dims=dims)
    xres, yres = calc_res(xarr)
    assert xres == 2
    assert yres == -1

    # x increasing, y decreasing
    xarr = DataArray(z, coords=dict(x=x, y=y[::-1]), dims=dims)
    xres, yres = calc_res(xarr)
    assert xres == 2
    assert yres == 1

    # x decreasing, y increasing
    xarr = DataArray(z, coords=dict(x=x[::-1], y=y), dims=dims)
    xres, yres = calc_res(xarr)
    assert xres == -2
    assert yres == -1

    # x and y decreasing
    xarr = DataArray(z, coords=dict(x=x[::-1], y=y[::-1]), dims=dims)
    xres, yres = calc_res(xarr)
    assert xres == -2
    assert yres == 1


def test_orient_array():
    x = [5, 7]
    y = [0, 1]
    z = np.array([[0, 1], [2, 3]])
    dims = ('y', 'x')

    # x and y increasing
    xarr = DataArray(z, coords=dict(x=x, y=y), dims=dims)
    arr = orient_array(xarr)
    assert np.array_equal(arr, z)

    # x increasing, y decreasing
    xarr = DataArray(z, coords=dict(x=x, y=y[::-1]), dims=dims)
    arr = orient_array(xarr)
    assert np.array_equal(arr, z[::-1])

    # x decreasing, y increasing
    xarr = DataArray(z, coords=dict(x=x[::-1], y=y), dims=dims)
    arr = orient_array(xarr)
    assert np.array_equal(arr, z[:, ::-1])

    # x and y decreasing
    xarr = DataArray(z, coords=dict(x=x[::-1], y=y[::-1]), dims=dims)
    arr = orient_array(xarr)
    assert np.array_equal(arr, z[::-1, ::-1])


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("Float64", "?float64"),
        ("Float32", "?float32"),
        ("Int64", "?int64"),
        ("Int32", "?int32"),
        ("float64[pyarrow]", "?float64"),
        ("int32[pyarrow]", "?int32"),
    ],
)
def test_dshape_from_pandas_masked(dtype, expected):
    df = pd.DataFrame({"v": pd.array([1, None, 3], dtype=dtype)})
    assert dshape_from_pandas(df).measure["v"] == dshape(expected).measure


def test_dshape_from_pandas_numpy_not_option():
    df = pd.DataFrame({"v": np.array([1.0, np.nan])})
    assert dshape_from_pandas(df).measure["v"] == dshape("float64").measure


def _masked_inputs():
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("nanoarrow")
    return {
        "np.ma": np.ma.masked_array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False]),
        "Float64": pd.Series([1.0, None, 3.0, 4.0], dtype="Float64"),
        "Int32": pd.Series([1, None, 3, 4], dtype="Int32"),
        "Float64 array": pd.array([1.0, None, 3.0, 4.0], dtype="Float64"),
        "arrow": pd.Series([1.0, None, 3.0, 4.0], dtype="float64[pyarrow]"),
        "arrow int": pd.Series([1, None, 3, 4], dtype="int32[pyarrow]"),
        "arrow sliced": pd.Series([0.0, 1.0, None, 3.0, 4.0], dtype="float64[pyarrow]")[1:],
        "arrow chunked": pd.Series(
            pa.chunked_array([[1.0], [None, 3.0], [4.0]]), dtype="float64[pyarrow]"
        ),
        "pa.Array": pa.array([1.0, None, 3.0, 4.0]),
        "pa.ChunkedArray sliced": pa.chunked_array([[0.0, 1.0, None], [3.0, 4.0]]).slice(1),
    }


@pytest.mark.parametrize(
    "name",
    [
        "np.ma", "Float64", "Int32", "Float64 array", "arrow", "arrow int", "arrow sliced",
        "arrow chunked", "pa.Array", "pa.ChunkedArray sliced",
    ],
)
def test_extract_values_and_mask(name):
    col = _masked_inputs()[name]
    values, mask = _extract_values_and_mask(col)
    assert isinstance(values, np.ndarray)
    assert len(values) == 4
    np.testing.assert_array_equal(mask, [True, False, True, True])
    np.testing.assert_array_equal(values[[0, 2, 3]], [1, 3, 4])


@pytest.mark.parametrize(
    "col",
    [
        np.array([1.0, np.nan]),
        np.ma.masked_array([1.0, 2.0]),
        pd.Series([1.0, 2.0], dtype="Float64"),
        pd.Series([1.0, 2.0], dtype="float64[pyarrow]"),
    ],
    ids=["numpy", "np.ma nomask", "Float64", "arrow"],
)
def test_extract_values_and_mask_no_nulls(col):
    pytest.importorskip("nanoarrow")
    values, mask = _extract_values_and_mask(col)
    assert mask is None
    np.testing.assert_array_equal(values, [1.0, col[1]])


def test_extract_values_and_mask_no_copy():
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("nanoarrow")
    masked = pd.array([1.0, None, 3.0], dtype="Float64")
    values, mask = _extract_values_and_mask(masked)
    assert np.shares_memory(values, masked._data)

    arrow = pa.array([1.0, None, 3.0])
    values, _ = _extract_values_and_mask(arrow)
    assert values.ctypes.data == arrow.buffers()[1].address
