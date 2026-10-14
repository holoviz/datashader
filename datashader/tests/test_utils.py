from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray

from datashader.datashape import dshape
from datashader.utils import (
    Dispatcher, _ArraySource, _column_values_and_mask, _extract_values_and_mask, apply, calc_res,
    dshape_from_pandas, isreal, orient_array,
)

try:
    import pyarrow as pa
except ImportError:
    pa = None

skip_no_pyarrow = pytest.mark.skipif(pa is None, reason="pyarrow is not installed")


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
        pytest.param("float64[pyarrow]", "?float64", marks=skip_no_pyarrow),
        pytest.param("int32[pyarrow]", "?int32", marks=skip_no_pyarrow),
    ],
)
def test_dshape_from_pandas_masked(dtype, expected):
    df = pd.DataFrame({"v": pd.array([1, None, 3], dtype=dtype)})
    assert dshape_from_pandas(df).measure["v"] == dshape(expected).measure


def test_dshape_from_pandas_numpy_not_option():
    df = pd.DataFrame({"v": np.array([1.0, np.nan])})
    assert dshape_from_pandas(df).measure["v"] == dshape("float64").measure


_masked_inputs = {
    "np.ma": lambda: np.ma.masked_array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False]),
    "Float64": lambda: pd.Series([1.0, None, 3.0, 4.0], dtype="Float64"),
    "Int32": lambda: pd.Series([1, None, 3, 4], dtype="Int32"),
    "Float64 array": lambda: pd.array([1.0, None, 3.0, 4.0], dtype="Float64"),
    "arrow": lambda: pd.Series([1.0, None, 3.0, 4.0], dtype="float64[pyarrow]"),
    "arrow int": lambda: pd.Series([1, None, 3, 4], dtype="int32[pyarrow]"),
    "arrow sliced": lambda: pd.Series([0.0, 1.0, None, 3.0, 4.0], dtype="float64[pyarrow]")[1:],
    "arrow chunked": lambda: pd.Series(
        pa.chunked_array([[1.0], [None, 3.0], [4.0]]), dtype="float64[pyarrow]"
    ),
    "pa.Array": lambda: pa.array([1.0, None, 3.0, 4.0]),
    "pa.ChunkedArray sliced": lambda: pa.chunked_array([[0.0, 1.0, None], [3.0, 4.0]]).slice(1),
}


@pytest.mark.parametrize(
    "name",
    [
        name if name in ("np.ma", "Float64", "Int32", "Float64 array")
        else pytest.param(name, marks=skip_no_pyarrow)
        for name in _masked_inputs
    ],
)
def test_extract_values_and_mask(name):
    values, mask = _extract_values_and_mask(_masked_inputs[name]())
    assert isinstance(values, np.ndarray)
    assert len(values) == 4
    np.testing.assert_array_equal(mask, [False, True, False, False])
    np.testing.assert_array_equal(values[[0, 2, 3]], [1, 3, 4])


@pytest.mark.parametrize(
    "make_col",
    [
        pytest.param(lambda: np.array([1.0, np.nan]), id="numpy"),
        pytest.param(lambda: np.ma.masked_array([1.0, 2.0]), id="np.ma nomask"),
        pytest.param(lambda: pd.Series([1.0, 2.0], dtype="Float64"), id="Float64"),
        pytest.param(
            lambda: pd.Series([1.0, 2.0], dtype="float64[pyarrow]"),
            id="arrow",
            marks=skip_no_pyarrow,
        ),
    ],
)
def test_extract_values_and_mask_no_nulls(make_col):
    col = make_col()
    values, mask = _extract_values_and_mask(col)
    assert mask is None
    np.testing.assert_array_equal(values, [1.0, col[1]])


def test_extract_values_and_mask_no_copy():
    masked = pd.array([1.0, None, 3.0], dtype="Float64")
    values, mask = _extract_values_and_mask(masked)
    assert np.shares_memory(values, masked._data)
    assert np.shares_memory(mask, masked._mask)


@skip_no_pyarrow
def test_extract_values_and_mask_arrow_no_copy():
    arrow = pa.array([1.0, None, 3.0])
    values, _ = _extract_values_and_mask(arrow)
    assert values.ctypes.data == arrow.buffers()[1].address


@skip_no_pyarrow
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("kind", ["timestamp", "duration"])
def test_extract_values_and_mask_arrow_temporal(kind, unit):
    arrow = pa.array([1, None, 3], type=getattr(pa, kind)(unit))
    values, mask = _extract_values_and_mask(arrow)
    np_kind = "datetime64" if kind == "timestamp" else "timedelta64"
    assert values.dtype == np.dtype(f"{np_kind}[{unit}]")
    np.testing.assert_array_equal(mask, [False, True, False])
    np.testing.assert_array_equal(values[[0, 2]].view("int64"), [1, 3])


def test_column_values_and_mask_series_no_copy():
    series = pd.Series([1.0, None, 3.0], dtype="Float64")
    values, mask = _column_values_and_mask(series)
    assert np.shares_memory(values, series.array._data)
    assert np.shares_memory(mask, series.array._mask)


@skip_no_pyarrow
def test_dshape_from_pandas_arrow_bool_not_option():
    df = pd.DataFrame({"v": pd.array([True, None], dtype="bool[pyarrow]")})
    assert dshape_from_pandas(df).measure["v"] == dshape("bool").measure


def test_array_source_wraps_numpy_without_copy():
    x = np.arange(5.0)
    assert np.shares_memory(_ArraySource({"x": x})["x"].to_numpy(), x)
