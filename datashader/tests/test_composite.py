from __future__ import annotations
import numpy as np

import pytest

from datashader import composite
from datashader.composite import (
    _add_arr_op, _add_op, _max_arr_op, _min_arr_op, _over_op, _saturate_op, _source_arr_op,
    _source_op, composite_op_lookup,
)

src = np.array([[0x00000000, 0x00ffffff, 0xffffffff],
                [0x7dff0000, 0x7d00ff00, 0x7d0000ff],
                [0xffff0000, 0xff000000, 0x3a3b3c3d]], dtype='uint32')

clear = np.uint32(0)
clear_white = np.uint32(0x00ffffff)
white = np.uint32(0xffffffff)
blue = np.uint32(0xffff0000)
half_blue = np.uint32(0x7dff0000)
half_purple = np.uint32(0x7d7d007d)


def test_source():
    o = src.copy()
    o[0, :2] = clear
    np.testing.assert_equal(_source_op(src, clear), o)
    o[0, :2] = clear_white
    np.testing.assert_equal(_source_op(src, clear_white), o)
    o[0, :2] = half_blue
    np.testing.assert_equal(_source_op(src, half_blue), o)


def test_over():
    o = src.copy()
    o[0, 1] = 0
    np.testing.assert_equal(_over_op(src, clear), o)
    np.testing.assert_equal(_over_op(src, clear_white), o)
    o = np.array([[0xffffffff, 0xffffffff, 0xffffffff],
                  [0xffff8282, 0xff82ff82, 0xff8282ff],
                  [0xffff0000, 0xff000000, 0xffd2d2d2]])
    np.testing.assert_equal(_over_op(src, white), o)
    o = np.array([[0xffff0000, 0xffff0000, 0xffffffff],
                  [0xffff0000, 0xff827d00, 0xff82007d],
                  [0xffff0000, 0xff000000, 0xffd20d0d]])
    np.testing.assert_equal(_over_op(src, blue), o)
    o = np.array([[0x7dff0000, 0x7dff0000, 0xffffffff],
                  [0xbcff0000, 0xbc56a800, 0xbc5600a8],
                  [0xffff0000, 0xff000000, 0x9ab51616]])
    np.testing.assert_equal(_over_op(src, half_blue), o)
    o = np.array([[0x7d7d007d, 0x7d7d007d, 0xffffffff],
                  [0xbcd3002a, 0xbc2aa82a, 0xbc2a00d3],
                  [0xffff0000, 0xff000000, 0x9a641664]])
    np.testing.assert_equal(_over_op(src, half_purple), o)


def test_add():
    o = src.copy()
    o[0, 1] = 0
    np.testing.assert_equal(_add_op(src, clear), o)
    np.testing.assert_equal(_add_op(src, clear_white), o)
    o = np.array([[0xffffffff, 0xffffffff, 0xffffffff],
                  [0xffffffff, 0xffffffff, 0xffffffff],
                  [0xffffffff, 0xffffffff, 0xffffffff]])
    np.testing.assert_equal(_add_op(src, white), o)
    o = np.array([[0xffff0000, 0xffff0000, 0xffffffff],
                  [0xffff0000, 0xffff7d00, 0xffff007d],
                  [0xffff0000, 0xffff0000, 0xffff0d0d]])
    np.testing.assert_equal(_add_op(src, blue), o)
    o = np.array([[0x7dff0000, 0x7dff0000, 0xffffffff],
                  [0xfaff0000, 0xfa7f7f00, 0xfa7f007f],
                  [0xffff0000, 0xff7d0000, 0xb7c01313]])
    np.testing.assert_equal(_add_op(src, half_blue), o)
    o = np.array([[0x7d7d007d, 0x7d7d007d, 0xffffffff],
                  [0xfabe003e, 0xfa3e7f3e, 0xfa3e00be],
                  [0xffff003d, 0xff3d003d, 0xb7681368]])
    np.testing.assert_equal(_add_op(src, half_purple), o)


def test_saturate():
    o = src.copy()
    o[0, 1] = 0
    np.testing.assert_equal(_saturate_op(src, clear), o)
    np.testing.assert_equal(_saturate_op(src, clear_white), o)
    o = np.full((3, 3), white, dtype='uint32')
    np.testing.assert_equal(_saturate_op(src, white), o)
    o = np.full((3, 3), blue, dtype='uint32')
    np.testing.assert_equal(_saturate_op(src, blue), o)
    o = np.array([[0x7dff0000, 0x7dff0000, 0xffff8282],
                  [0xfaff0000, 0xfa7f7f00, 0xfa7f007f],
                  [0xffff0000, 0xff7d0000, 0xb7c01313]])
    np.testing.assert_equal(_saturate_op(src, half_blue), o)
    o = np.array([[0x7d7d007d, 0x7d7d007d, 0xffbf82bf],
                  [0xfabe003e, 0xfa3e7f3e, 0xfa3e00be],
                  [0xffbf003d, 0xff3d003d, 0xb7681368]])
    np.testing.assert_equal(_saturate_op(src, half_purple), o)


image_ops = [_over_op, _add_op, _saturate_op, _source_op]


@pytest.mark.parametrize("op", image_ops)
@pytest.mark.parametrize(
    "d",
    [np.array([1, 2], "uint8"), np.array([1, 2], "uint16"), np.array([True, False]), 5],
    ids=["uint8", "uint16", "bool", "pyint"],
)
def test_image_operators_safe_cast(op, d):
    s = np.array([0x7dff0000, 0xff00ff00], dtype="uint32")
    out = op(s, d)
    assert out.dtype == np.uint32
    np.testing.assert_equal(out, op(s, np.asarray(d, dtype="uint32")))


@pytest.mark.parametrize("op", image_ops)
@pytest.mark.parametrize(
    "d",
    [np.array([1, 2], "int32"), np.array([1.0, np.nan]), 1.5, [1, 2]],
    ids=["int32", "float64", "pyfloat", "list"],
)
def test_image_operators_unsafe_cast(op, d):
    s = np.array([0x7dff0000, 0xff00ff00], dtype="uint32")
    with pytest.raises(TypeError, match="Unsupported dtype for image composite operators"):
        op(s, d)


@pytest.mark.parametrize("op", image_ops)
def test_image_operators_python_ints(op):
    with pytest.raises(TypeError, match="Unsupported dtype for image composite operators"):
        op(1, 2)


arr_refs = {
    _add_arr_op: np.add,
    _max_arr_op: np.maximum,
    _min_arr_op: np.minimum,
    _source_arr_op: lambda s, d: np.where(s != 0, s, d),
}


def _arr_ref(op, s, d):
    ref = arr_refs[op](s, d)
    return ref.astype("int64") if op is _add_arr_op and ref.dtype == "int32" else ref


@pytest.mark.parametrize("op", arr_refs)
@pytest.mark.parametrize(
    ("dtype", "loop_dtype"),
    [
        ("bool", "int32"),
        ("int8", "int32"),
        ("int32", "int32"),
        ("int64", "int64"),
        ("uint64", "float64"),
        ("float16", "float32"),
        ("float32", "float32"),
        ("float64", "float64"),
    ],
)
def test_array_operators_dtype(op, dtype, loop_dtype):
    s = np.array([[0, 1, 5], [3, 0, 2]], dtype=dtype)
    d = np.array([[4, 0, 1], [3, 2, 7]], dtype=dtype)
    out = op(s, d)
    ref = _arr_ref(op, s.astype(loop_dtype), d.astype(loop_dtype))
    assert out.dtype == ref.dtype
    np.testing.assert_equal(out, ref)


@pytest.mark.parametrize("op", arr_refs)
@pytest.mark.parametrize(
    ("s", "d"),
    [
        (np.array([0, 1, 5], dtype="int32"), 3),
        (np.array([0, 1, 5], dtype="float32"), 3.5),
    ],
    ids=["int32-pyint", "float32-pyfloat"],
)
def test_array_operators_python_scalar(op, s, d):
    out = op(s, d)
    ref = _arr_ref(op, s, np.asarray(d, dtype=s.dtype))
    assert out.dtype == ref.dtype
    np.testing.assert_equal(out, ref)


@pytest.mark.parametrize("op", arr_refs)
def test_array_operators_list(op):
    out = op([0, 1, 5], [4, 0, 1])
    ref = _arr_ref(op, np.array([0, 1, 5]), np.array([4, 0, 1]))
    assert out.dtype == ref.dtype
    np.testing.assert_equal(out, ref)


@pytest.mark.parametrize("op", arr_refs)
def test_array_operators_broadcast(op):
    s = np.array([[0, 1, 5], [3, 0, 2]], dtype="float64")
    d = np.array([4, 0, 1], dtype="float64")
    np.testing.assert_equal(op(s, d), arr_refs[op](s, np.broadcast_to(d, s.shape)))


@pytest.mark.parametrize("op", arr_refs)
def test_array_operators_scalar_and_empty(op):
    out = op(np.float64(0), np.float64(2))
    assert isinstance(out, np.float64)
    assert out == arr_refs[op](np.float64(0), np.float64(2))
    empty = np.empty((0, 3), dtype="float32")
    assert op(empty, empty).shape == (0, 3)


def test_composite_op_lookup():
    assert composite_op_lookup == {
        "over": _over_op, "add": _add_op, "saturate": _saturate_op, "source": _source_op,
        "add_arr": _add_arr_op, "max_arr": _max_arr_op, "min_arr": _min_arr_op,
        "source_arr": _source_arr_op,
    }


@pytest.mark.parametrize("op", [_over_op, _add_arr_op, _max_arr_op])
def test_operators_stay_lazy_on_dask(op):
    dask = pytest.importorskip("dask")
    da = pytest.importorskip("dask.array")
    dtype = "uint32" if op is _over_op else "float64"
    s = np.array([[0, 0x7dff0000, 5], [3, 0, 0xff000000]], dtype=dtype)
    d = np.array([0xffffffff, 0, 1], dtype=dtype)
    with dask.config.set(scheduler=_raise_on_compute):
        out = op(da.from_array(s, chunks=1), da.from_array(d, chunks=2))
    assert isinstance(out, da.Array)
    expected = op(s, d)
    assert out.dtype == expected.dtype
    np.testing.assert_equal(out.compute(), expected)


def _raise_on_compute(*args, **kwargs):
    raise AssertionError("dask array was computed")


@pytest.mark.parametrize("name", composite.image_operators + composite.array_operators)
def test_deprecated_operator_ufuncs(name):
    with pytest.warns(FutureWarning, match=rf"'datashader\.composite\.{name}' is deprecated"):
        ufunc = getattr(composite, name)
    dtype = "uint32" if name in composite.image_operators else "float64"
    s = src.astype(dtype)
    d = np.array([white, 0, blue], dtype=dtype)
    np.testing.assert_equal(ufunc(s, d), composite_op_lookup[name](s, d))
    assert ufunc.__name__ == name
    if name in composite.array_operators:
        np.testing.assert_equal(ufunc(s.tolist(), d.tolist()), composite_op_lookup[name](s, d))


def test_deprecated_operator_decorators(monkeypatch):
    monkeypatch.setattr(composite, "composite_op_lookup", dict(composite_op_lookup))
    with pytest.warns(FutureWarning, match=r"'datashader\.composite\.operator' is deprecated"):
        @composite.operator
        def keep_src(src, dst):
            return src

    with pytest.warns(FutureWarning, match=r"'datashader\.composite\.arr_operator' is deprecated"):
        @composite.arr_operator
        def keep_dst(src, dst):
            return dst

    np.testing.assert_equal(keep_src(src, white), src)
    d = np.arange(3.0)
    np.testing.assert_equal(keep_dst(d + 1, d), d)
    assert composite.composite_op_lookup["keep_src"] is keep_src
    assert composite.composite_op_lookup["keep_dst"] is keep_dst


def test_unknown_attribute():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        composite.nope
