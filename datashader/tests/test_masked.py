from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import pytest

import datashader as ds
import datashader.utils as du
from datashader.tests.test_pandas import assert_eq_xr

try:
    import pyarrow as pa
except ImportError:
    pa = None

try:
    import polars as pl
except ImportError:
    pl = None


skip_no_pyarrow = pytest.mark.skipif(pa is None, reason="pyarrow is not installed")
skip_no_polars = pytest.mark.skipif(pl is None, reason="polars is not installed")
_ARROW_FLOAT = pytest.param("float64[pyarrow]", marks=skip_no_pyarrow)
_ARROW_INT = pytest.param("int32[pyarrow]", marks=skip_no_pyarrow)

_masked_reductions = [
    ds.count(), ds.count("v"), ds.any("v"), ds.sum("v"), ds.min("v"), ds.max("v"),
    ds.mean("v"), ds.var("v"), ds.std("v"), ds.first("v"), ds.last("v"),
    ds.first_n("v", n=2), ds.last_n("v", n=2), ds.max_n("v", n=2), ds.min_n("v", n=2),
]


def _masked_frames(dtype, n=200, seed=1):
    rng = np.random.default_rng(seed)
    data = {
        "x": rng.uniform(0, 1, n),
        "y": rng.uniform(0, 1, n),
        "v": rng.integers(-50, 50, n).astype(float),
    }
    nulls = {k: rng.random(n) < 0.2 for k in data}
    nan_df = pd.DataFrame({k: np.where(nulls[k], np.nan, v) for k, v in data.items()})
    return nan_df, _to_masked(nan_df, dtype)


def _to_masked(nan_df, dtype):
    # 1e6 behind the mask makes a test fail if the mask is ignored
    dt = pd.api.types.pandas_dtype(dtype)
    out = {}
    for name, col in nan_df.items():
        null = col.isna().to_numpy()
        data = np.where(null, 1e6, col.to_numpy()).astype(dt.numpy_dtype)
        if isinstance(dt, pd.ArrowDtype):
            arr = pd.arrays.ArrowExtensionArray(pa.array(data, type=dt.pyarrow_dtype, mask=null))
        else:
            arr = type(pd.array([], dtype=dt))(data, null)
        out[name] = pd.Series(arr, index=nan_df.index)
    return pd.DataFrame(out)


@pytest.mark.parametrize("dtype", ["Float64", _ARROW_FLOAT])
@pytest.mark.parametrize("reduction", _masked_reductions, ids=repr)
def test_points_masked(dtype, reduction):
    nan_df, masked_df = _masked_frames(dtype)
    cvs = ds.Canvas(plot_width=7, plot_height=5, x_range=(0, 1), y_range=(0, 1))
    expected = cvs.points(nan_df, "x", "y", reduction)
    actual = cvs.points(masked_df, "x", "y", reduction)
    assert_eq_xr(actual, expected)


@pytest.mark.parametrize("dtype", ["Int64", _ARROW_INT])
@pytest.mark.parametrize("reduction", [ds.count("v"), ds.sum("v"), ds.max("v")], ids=repr)
def test_points_masked_int(dtype, reduction):
    nan_df, _ = _masked_frames("Float64")
    masked_df = nan_df.assign(v=_to_masked(nan_df[["v"]], dtype)["v"])
    cvs = ds.Canvas(plot_width=7, plot_height=5, x_range=(0, 1), y_range=(0, 1))
    expected = cvs.points(nan_df, "x", "y", reduction)
    actual = cvs.points(masked_df, "x", "y", reduction)
    np.testing.assert_array_equal(actual.values.astype(float), expected.values)


@pytest.mark.parametrize("dtype", ["Float64", _ARROW_FLOAT])
def test_points_masked_sliced(dtype):
    nan_df, masked_df = _masked_frames(dtype)
    cvs = ds.Canvas(plot_width=7, plot_height=5, x_range=(0, 1), y_range=(0, 1))
    expected = cvs.points(nan_df.iloc[13:], "x", "y", ds.sum("v"))
    actual = cvs.points(masked_df.iloc[13:], "x", "y", ds.sum("v"))
    assert_eq_xr(actual, expected)


@skip_no_pyarrow
def test_points_masked_arrow_chunks():
    nan_df, masked_df = _masked_frames("float64[pyarrow]")
    chunked = pd.DataFrame({
        k: pd.Series(pa.chunked_array([arr[:50], arr[50:51], arr[51:]]), dtype="float64[pyarrow]")
        for k, arr in ((k, pa.array(masked_df[k])) for k in masked_df.columns)
    })
    cvs = ds.Canvas(plot_width=7, plot_height=5, x_range=(0, 1), y_range=(0, 1))
    for reduction in [ds.sum("v"), ds.first("v"), ds.last("v")]:
        assert_eq_xr(
            cvs.points(chunked, "x", "y", reduction), cvs.points(nan_df, "x", "y", reduction)
        )


@pytest.mark.parametrize("dtype", ["Float64", _ARROW_FLOAT])
def test_points_masked_no_nulls(dtype):
    nan_df, _ = _masked_frames(dtype)
    nan_df = nan_df.fillna(0.5)
    masked_df = nan_df.astype(dtype)
    cvs = ds.Canvas(plot_width=7, plot_height=5, x_range=(0, 1), y_range=(0, 1))
    assert_eq_xr(
        cvs.points(masked_df, "x", "y", ds.sum("v")), cvs.points(nan_df, "x", "y", ds.sum("v"))
    )


@pytest.mark.parametrize("dtype", ["Float64", _ARROW_FLOAT])
def test_points_masked_auto_bounds(dtype):
    nan_df, masked_df = _masked_frames(dtype)
    cvs = ds.Canvas(plot_width=7, plot_height=5)
    expected = cvs.points(nan_df, "x", "y", ds.sum("v"))
    actual = cvs.points(masked_df, "x", "y", ds.sum("v"))
    assert_eq_xr(actual, expected)
    assert actual.attrs["x_range"] == expected.attrs["x_range"]
    assert actual.attrs["y_range"] == expected.attrs["y_range"]


@pytest.mark.parametrize("dtype", ["Float64", _ARROW_FLOAT])
@pytest.mark.parametrize(
    "reduction",
    [
        ds.where(ds.max("v")),
        ds.where(ds.min("v"), "w"),
        ds.where(ds.first("v"), "w"),
        ds.where(ds.max_n("v", n=2), "w"),
        ds.where(ds.max("x"), "w"),
        ds.by("cat", ds.where(ds.max("v"), "w")),
        ds.summary(a=ds.where(ds.max("v"), "w"), b=ds.where(ds.max("v"), "y")),
    ],
    ids=repr,
)
def test_points_masked_where(dtype, reduction):
    nan_df, masked_df = _masked_frames(dtype)
    _, masked_w = _masked_frames(dtype, seed=2)
    nan_w, _ = _masked_frames(dtype, seed=2)
    nan_df["w"], masked_df["w"] = nan_w["v"], masked_w["v"]
    nan_df["cat"] = masked_df["cat"] = pd.Categorical(np.arange(len(nan_df)) % 3)
    cvs = ds.Canvas(plot_width=7, plot_height=5, x_range=(0, 1), y_range=(0, 1))
    expected = cvs.points(nan_df, "x", "y", reduction)
    actual = cvs.points(masked_df, "x", "y", reduction)
    xr.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["Float64", _ARROW_FLOAT, "Int64", _ARROW_INT])
@pytest.mark.parametrize("reduction", [ds.count("v"), ds.sum("v"), ds.max("v")], ids=repr)
def test_points_masked_by(dtype, reduction):
    nan_df, _ = _masked_frames("Float64")
    nan_df["cat"] = pd.Categorical(np.arange(len(nan_df)) % 3)
    masked_df = nan_df.assign(v=_to_masked(nan_df[["v"]], dtype)["v"])
    cvs = ds.Canvas(plot_width=7, plot_height=5, x_range=(0, 1), y_range=(0, 1))
    expected = cvs.points(nan_df, "x", "y", ds.by("cat", reduction))
    actual = cvs.points(masked_df, "x", "y", ds.by("cat", reduction))
    assert_eq_xr(actual, expected)


_masked_line_specs = {
    "LineAxis0": dict(x="x0", y="y0", axis=0),
    "LineAxis0Multi": dict(x=["x0", "x1"], y=["y0", "y1"], axis=0),
    "LinesAxis1": dict(x=["x0", "x1", "x2"], y=["y0", "y1", "y2"], axis=1),
    "LinesAxis1XConstant": dict(x=np.array([0.1, 0.5, 0.9]), y=["y0", "y1", "y2"], axis=1),
    "LinesAxis1YConstant": dict(x=["x0", "x1", "x2"], y=np.array([0.1, 0.5, 0.9]), axis=1),
}


@pytest.mark.parametrize("dtype", ["Float64", _ARROW_FLOAT, "Int64"])
@pytest.mark.parametrize("spec", list(_masked_line_specs))
def test_line_masked(dtype, spec):
    rng = np.random.default_rng(3)
    n = 30
    scale = 100 if dtype == "Int64" else 1
    data = {f"{k}{j}": np.round(rng.uniform(0, 1, n) * scale) for k in "xy" for j in range(3)}
    data["v"] = rng.integers(0, 10, n).astype(float)
    nan_df = pd.DataFrame({
        k: np.where(rng.random(n) < 0.2, np.nan, v) for k, v in data.items()
    })
    masked_df = _to_masked(nan_df, dtype)
    rng_ = (0, scale)
    cvs = ds.Canvas(plot_width=9, plot_height=7, x_range=rng_, y_range=rng_)
    kwargs = _masked_line_specs[spec]
    if scale != 1:
        kwargs = {k: v * scale if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
    for agg in [ds.count(), ds.sum("v")]:
        expected = cvs.line(nan_df, agg=agg, **kwargs)
        actual = cvs.line(masked_df, agg=agg, **kwargs)
        assert_eq_xr(actual, expected)
    actual = ds.Canvas(plot_width=9, plot_height=7).line(masked_df, **kwargs)
    expected = ds.Canvas(plot_width=9, plot_height=7).line(nan_df, **kwargs)
    assert_eq_xr(actual, expected)


def _dict_source(kind, nan_df):
    out = {}
    for k, v in nan_df.items():
        v = v.to_numpy()
        missing = np.isnan(v)
        if kind == "numpy":
            out[k] = v
        elif kind == "np.ma":
            out[k] = np.ma.masked_array(np.where(missing, 1e6, v), mask=missing)
        elif kind == "Float64":
            out[k] = pd.arrays.FloatingArray(np.where(missing, 1e6, v), missing)
        elif kind == "arrow pd.array":
            out[k] = pd.array(np.where(missing, None, v).tolist(), dtype="float64[pyarrow]")
        elif kind == "arrow":
            out[k] = pa.array(v, mask=missing)
        elif kind == "arrow chunked":
            arr = pa.array(v, mask=missing)
            out[k] = pa.chunked_array([arr[:77], arr[77:]])
        elif kind == "polars":
            out[k] = pl.Series(k, v, nan_to_null=True)
    return out


_dict_kinds = [
    "numpy", "np.ma", "Float64",
    pytest.param("arrow", marks=skip_no_pyarrow),
    pytest.param("arrow chunked", marks=skip_no_pyarrow),
    pytest.param("arrow pd.array", marks=skip_no_pyarrow),
    pytest.param("polars", marks=skip_no_polars),
]


@pytest.mark.parametrize("kind", _dict_kinds)
@pytest.mark.parametrize("reduction", [ds.count(), ds.sum("v"), ds.first("v")], ids=repr)
def test_points_dict_source(kind, reduction):
    nan_df, _ = _masked_frames("Float64")
    source = _dict_source(kind, nan_df)
    source["unused"] = np.zeros((3, 3))
    for cvs in [
        ds.Canvas(plot_width=7, plot_height=5, x_range=(0, 1), y_range=(0, 1)),
        ds.Canvas(plot_width=7, plot_height=5),
    ]:
        assert_eq_xr(
            cvs.points(source, "x", "y", reduction), cvs.points(nan_df, "x", "y", reduction)
        )


@pytest.mark.parametrize("kind", _dict_kinds)
@pytest.mark.parametrize("spec", list(_masked_line_specs))
def test_line_dict_source(kind, spec):
    rng = np.random.default_rng(3)
    n = 30
    nan_df = pd.DataFrame({
        f"{k}{j}": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(0, 1, n))
        for k in "xy" for j in range(3)
    })
    source = _dict_source(kind, nan_df)
    cvs = ds.Canvas(plot_width=9, plot_height=7)
    kwargs = _masked_line_specs[spec]
    assert_eq_xr(cvs.line(source, **kwargs), cvs.line(nan_df, **kwargs))


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ({"x": np.zeros((2, 2)), "y": np.zeros(2)}, "1D"),
        ({"x": np.zeros(3), "y": np.zeros(2)}, "same length"),
    ],
)
def test_dict_source_invalid(source, match):
    cvs = ds.Canvas(plot_width=7, plot_height=5)
    with pytest.raises(ValueError, match=match):
        cvs.points(source, "x", "y")


def test_dict_source_unsupported_glyph():
    cvs = ds.Canvas(plot_width=7, plot_height=5)
    with pytest.raises(NotImplementedError, match="dict"):
        cvs.line({"x": np.zeros(3), "y": np.zeros(3)}, "x", "y", axis=1)


def test_points_dict_source_lists():
    source = {"x": [0.1, 0.6, 0.9], "y": [0.1, 0.6, 0.2], "v": [1, 2, 3]}
    cvs = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))
    assert_eq_xr(
        cvs.points(source, "x", "y", ds.sum("v")),
        cvs.points(pd.DataFrame(source), "x", "y", ds.sum("v")),
    )


def test_dict_source_missing_column():
    cvs = ds.Canvas(plot_width=7, plot_height=5)
    with pytest.raises(ValueError, match="'y'"):
        cvs.points({"x": np.zeros(3)}, "x", "y")


_c3 = np.array([0.1, 0.5, 0.9])
_masked_area_specs = {
    "AreaToZeroAxis0": dict(x="x0", y="y0", axis=0),
    "AreaToLineAxis0": dict(x="x0", y="y0", y_stack="y1", axis=0),
    "AreaToZeroAxis0Multi": dict(x=["x0", "x1"], y=["y0", "y1"], axis=0),
    "AreaToLineAxis0Multi": dict(x=["x0", "x1"], y=["y0", "y1"], y_stack=["y2", "y0"], axis=0),
    "AreaToZeroAxis1": dict(x=["x0", "x1", "x2"], y=["y0", "y1", "y2"], axis=1),
    "AreaToLineAxis1": dict(
        x=["x0", "x1", "x2"], y=["y0", "y1", "y2"], y_stack=["y2", "y0", "y1"], axis=1
    ),
    "AreaToZeroAxis1XConstant": dict(x=_c3, y=["y0", "y1", "y2"], axis=1),
    "AreaToLineAxis1XConstant": dict(
        x=_c3, y=["y0", "y1", "y2"], y_stack=["y2", "y0", "y1"], axis=1
    ),
    "AreaToZeroAxis1YConstant": dict(x=["x0", "x1", "x2"], y=_c3, axis=1),
    "AreaToLineAxis1YConstant": dict(x=["x0", "x1", "x2"], y=_c3, y_stack=_c3[::-1], axis=1),
}


@pytest.mark.parametrize("source", ["Float64", _ARROW_FLOAT, "dict np.ma"])
@pytest.mark.parametrize("spec", list(_masked_area_specs))
def test_area_masked(source, spec):
    rng = np.random.default_rng(4)
    n = 30
    nan_df = pd.DataFrame({
        f"{k}{j}": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(0, 1, n))
        for k in "xy" for j in range(3)
    })
    if source == "dict np.ma":
        masked = _dict_source("np.ma", nan_df)
    else:
        masked = _to_masked(nan_df, source)
    cvs = ds.Canvas(plot_width=9, plot_height=7)
    kwargs = _masked_area_specs[spec]
    assert_eq_xr(cvs.area(masked, **kwargs), cvs.area(nan_df, **kwargs))


@pytest.mark.parametrize("kind", _dict_kinds)
@pytest.mark.parametrize(
    "reduction",
    [
        ds.by("cat", ds.count()),
        ds.by("cat", ds.sum("v")),
        ds.by(ds.category_binning("v", 0, 50, 5), ds.count()),
        ds.by(ds.category_modulo("i", 3), ds.max("v")),
    ],
    ids=repr,
)
def test_points_dict_source_by(kind, reduction):
    nan_df, _ = _masked_frames("Float64")
    source = _dict_source(kind, nan_df)
    source["cat"] = pd.Categorical(np.arange(len(nan_df)) % 3)
    source["i"] = np.arange(len(nan_df))
    df = nan_df.assign(cat=source["cat"], i=source["i"])
    cvs = ds.Canvas(plot_width=7, plot_height=5, x_range=(0, 1), y_range=(0, 1))
    assert_eq_xr(cvs.points(source, "x", "y", reduction), cvs.points(df, "x", "y", reduction))


@pytest.mark.parametrize("where", ["vertices", "simplices", "mesh"])
def test_trimesh_masked_raises(where):
    verts = pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0]})
    tris = pd.DataFrame({"v0": [0, 1], "v1": [1, 3], "v2": [2, 2]})
    mesh = None
    if where == "vertices":
        verts["z"] = pd.array([1.0, None, 3.0, 4.0], dtype="Float64")
    elif where == "simplices":
        tris["w"] = pd.array([1.0, None], dtype="Float64")
    else:
        tris["w"] = [1.0, 2.0]
        mesh = du.mesh(verts, tris)
        mesh["w"] = pd.array([1.0, 1.0, 1.0, None, None, None], dtype="Float64")
    cvs = ds.Canvas(plot_width=5, plot_height=5)
    with pytest.raises(NotImplementedError, match="Triangles" if mesh is not None else "trimesh"):
        cvs.trimesh(verts, tris, mesh=mesh)


def _narwhals_source(backend, nan_df):
    nw = pytest.importorskip("narwhals")
    if backend == "pandas Float64":
        native = nan_df.astype("Float64")
    elif backend == "pandas arrow":
        native = nan_df.astype("float64[pyarrow]")
    elif backend == "pyarrow":
        native = pa.table({k: pa.array(v.to_numpy(), from_pandas=True) for k, v in nan_df.items()})
    elif backend == "polars":
        native = pl.from_pandas(nan_df, nan_to_null=True)
    return nw.from_native(native, eager_only=True)


_narwhals_backends = [
    pytest.param("pandas Float64", marks=skip_no_pyarrow),
    pytest.param("pandas arrow", marks=skip_no_pyarrow),
    pytest.param("pyarrow", marks=skip_no_pyarrow),
    pytest.param("polars", marks=skip_no_polars),
]


@pytest.mark.parametrize("native", [False, True], ids=["narwhals", "native"])
@pytest.mark.parametrize("backend", _narwhals_backends)
@pytest.mark.parametrize("reduction", [ds.count(), ds.sum("v"), ds.first("v")], ids=repr)
def test_points_narwhals(backend, reduction, native):
    nan_df, _ = _masked_frames("Float64")
    source = _narwhals_source(backend, nan_df)
    if native:
        source = source.to_native()
    for cvs in [
        ds.Canvas(plot_width=7, plot_height=5, x_range=(0, 1), y_range=(0, 1)),
        ds.Canvas(plot_width=7, plot_height=5),
    ]:
        assert_eq_xr(
            cvs.points(source, "x", "y", reduction), cvs.points(nan_df, "x", "y", reduction)
        )


@pytest.mark.parametrize("backend", _narwhals_backends)
def test_line_narwhals(backend):
    rng = np.random.default_rng(3)
    n = 30
    nan_df = pd.DataFrame({
        f"{k}{j}": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(0, 1, n))
        for k in "xy" for j in range(3)
    })
    source = _narwhals_source(backend, nan_df)
    cvs = ds.Canvas(plot_width=9, plot_height=7)
    kwargs = _masked_line_specs["LinesAxis1"]
    assert_eq_xr(cvs.line(source, **kwargs), cvs.line(nan_df, **kwargs))


@skip_no_pyarrow
def test_arrow_stream_source_not_table():
    cvs = ds.Canvas(plot_width=7, plot_height=5)
    with pytest.raises(ValueError, match="table"):
        cvs.points(pa.chunked_array([[1.0, 2.0]]), "x", "y")


_CAT_VALUES = ["b", "a", None, "c", "a", "b", None, "c", "b", "a"] * 20


def _categorical_source(kind, values):
    if kind == "pd.Categorical":
        return pd.Categorical(values)
    elif kind == "pandas arrow":
        return pd.Series(pa.array(values).dictionary_encode(), dtype=pd.ArrowDtype(
            pa.dictionary(pa.int32(), pa.string())))
    elif kind == "arrow":
        return pa.array(values).dictionary_encode()
    elif kind == "arrow chunked":
        # Each chunk gets its own dictionary
        return pa.chunked_array([
            pa.array(values[:77]).dictionary_encode(), pa.array(values[77:]).dictionary_encode()
        ])
    elif kind == "polars":
        return pl.Series(values, dtype=pl.Categorical)
    elif kind == "polars sliced":
        return pl.Series([None, "c", *values], dtype=pl.Categorical)[2:]
    elif kind == "polars Enum":
        return pl.Series(values, dtype=pl.Enum(["c", "d", "b", "a"]))


_categorical_kinds = [
    "pd.Categorical",
    pytest.param("pandas arrow", marks=skip_no_pyarrow),
    pytest.param("arrow", marks=skip_no_pyarrow),
    pytest.param("arrow chunked", marks=skip_no_pyarrow),
    pytest.param("polars", marks=skip_no_polars),
    pytest.param("polars sliced", marks=skip_no_polars),
    pytest.param("polars Enum", marks=skip_no_polars),
]


@pytest.mark.parametrize("kind", _categorical_kinds)
@pytest.mark.parametrize("reduction", [ds.count(), ds.sum("v"), ds.first("v")], ids=repr)
def test_points_by_categorical(kind, reduction):
    nan_df, _ = _masked_frames("Float64", n=len(_CAT_VALUES))
    source = {**_dict_source("numpy", nan_df), "cat": _categorical_source(kind, _CAT_VALUES)}
    cvs = ds.Canvas(plot_width=7, plot_height=5, x_range=(0, 1), y_range=(0, 1))
    actual = cvs.points(source, "x", "y", ds.by("cat", reduction))
    categories = list(actual.coords["cat"].values)
    expected_df = nan_df.assign(cat=pd.Categorical(_CAT_VALUES, categories=categories))
    expected = cvs.points(expected_df, "x", "y", ds.by("cat", reduction))
    assert_eq_xr(actual, expected)
    assert set(_CAT_VALUES) - {None} <= set(categories)


@pytest.mark.parametrize("kind", [
    pytest.param("pandas arrow", marks=skip_no_pyarrow),
    pytest.param("polars Enum", marks=skip_no_polars),
])
def test_points_by_categorical_dataframe(kind):
    nan_df, _ = _masked_frames("Float64", n=len(_CAT_VALUES))
    cat = _categorical_source(kind, _CAT_VALUES)
    if kind == "pandas arrow":
        source = nan_df.assign(cat=cat)
    else:
        source = pl.from_pandas(nan_df, nan_to_null=True).with_columns(cat=cat)
    cvs = ds.Canvas(plot_width=7, plot_height=5, x_range=(0, 1), y_range=(0, 1))
    actual = cvs.points(source, "x", "y", ds.by("cat", ds.count()))
    categories = list(actual.coords["cat"].values)
    expected_df = nan_df.assign(cat=pd.Categorical(_CAT_VALUES, categories=categories))
    assert_eq_xr(actual, cvs.points(expected_df, "x", "y", ds.by("cat", ds.count())))


@skip_no_pyarrow
def test_categorical_arrow_chunked_categories():
    source = pa.chunked_array([
        pa.array(["b", "a"]).dictionary_encode(), pa.array(["c", "a"]).dictionary_encode()
    ])
    dshape = du._dshape_from_array_source(du._ArraySource({"cat": source}))
    assert list(dshape.measure["cat"].categories) == ["b", "a", "c"]


@skip_no_polars
def test_categorical_polars_enum_categories():
    source = pl.Series(["a", None], dtype=pl.Enum(["c", "d", "b", "a"]))
    dshape = du._dshape_from_array_source(du._ArraySource({"cat": source}))
    assert list(dshape.measure["cat"].categories) == ["c", "d", "b", "a"]


@skip_no_pyarrow
def test_arrow_categorical_is_not_masked():
    assert not du._is_masked(pa.array(["a", "b"]).dictionary_encode())


@skip_no_polars
@pytest.mark.parametrize("glyph, agg, column, arrow_type", [
    ("points", ds.count(), pl.Series([1, 2], dtype=pl.Datetime), "timestamp"),
    ("points", ds.by("c"), pl.Series(["a", "b"]), "string_view"),
    ("points", ds.any("c"), pl.Series([True, False]), "bool"),
    ("points", ds.max("c"), pl.Series([[1], [2]]), "large_list"),
    ("line", ds.count(), pl.Series([1, 2], dtype=pl.Datetime), "timestamp"),
] if pl is not None else [])
def test_unsupported_arrow_column(glyph, agg, column, arrow_type):
    name = "x" if agg.column is None else "c"
    source = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}).with_columns(column.alias(name))
    cvs = ds.Canvas(plot_width=2, plot_height=2)
    match = f"column {name!r} has unsupported Arrow type {arrow_type}"
    with pytest.raises(NotImplementedError, match=match):
        getattr(cvs, glyph)(source, "x", "y", agg=agg)


@skip_no_pyarrow
def test_unsupported_arrow_half_float():
    source = {"x": pa.array([0.0, 1.0], pa.float16()), "y": np.array([0.0, 1.0])}
    cvs = ds.Canvas(plot_width=2, plot_height=2)
    match = "column 'x' has unsupported Arrow type half_float"
    with pytest.raises(NotImplementedError, match=match):
        cvs.points(source, "x", "y")
