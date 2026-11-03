import numpy as np
import pandas as pd
import pytest
import xarray as xr

import datashader as ds


def all_subclasses(cls):
    items1 = {cls, *cls.__subclasses__()}
    items2 = {s for c in cls.__subclasses__() for s in all_subclasses(c)}
    return items1 | items2


def test_string_output():
    expected = {
        "any": "any('col')",
        "by": "by(column='col', reduction=count())",
        "count": "count()",
        "count_cat": "count_cat(column='col')",
        "first": "first('col')",
        "first_n": "first_n(column='col', n=1)",
        "last": "last('col')",
        "last_n": "last_n(column='col', n=1)",
        "m2": "m2('col')",
        "max": "max('col')",
        "max_n": "max_n(column='col', n=1)",
        "mean": "mean('col')",
        "min": "min('col')",
        "min_n": "min_n(column='col', n=1)",
        "mode": "mode('col')",
        "std": "std('col')",
        "sum": "sum('col')",
        "summary": "summary(a=1)",
        "var": "var('col')",
        "where": "where(selector=min('col'), lookup_column='col')",
    }

    count = 0
    for red in all_subclasses(ds.reductions.Reduction) | all_subclasses(ds.reductions.summary):
        red_name = red.__name__
        if red_name.startswith("_") or "Reduction" in red_name:
            continue
        elif red_name in ("by", "count_cat"):
            assert str(red("col")) == expected[red_name]
        elif red_name == "where":
            assert str(red(ds.min("col"), "col")) == expected[red_name]
        elif red_name == "summary":
            assert str(red(a=1)) == expected[red_name]
        else:
            assert str(red("col")) == expected[red_name]
        count += 1

    assert count == 20  # Update if more subclasses are added


def test_mode_raise_error():
    # Test for https://github.com/holoviz/datashader/issues/1435
    xr_ds = xr.Dataset(
        {"foo": (("x", "y"), np.arange(120).reshape(30, 4))},
        coords={"x": np.arange(30), "y": np.arange(4)},
    )
    xr_ds = xr_ds.drop_indexes(("x", "y"))
    xr_ds["x"], xr_ds["y"] = xr.broadcast(xr_ds.x, xr_ds.y)

    cvs = ds.Canvas(x_range=(0, 3), y_range=(0, 4))
    with pytest.raises(NotImplementedError):
        cvs.quadmesh(xr_ds, x="x", y="y", agg=ds.reductions.mode("foo"))


def _compile_components_for(df, agg, antialias=False):
    from datashader.compiler import compile_components
    from datashader.utils import dshape_from_pandas

    glyph = ds.glyphs.LineAxis0("x", "y") if antialias else ds.glyphs.Point("x", "y")
    if antialias:
        glyph.set_line_width(1)
    schema = dshape_from_pandas(df).measure
    return compile_components(agg, schema, glyph, antialias=antialias)


@pytest.mark.parametrize("antialias", [False, True])
@pytest.mark.parametrize("reduction", [ds.max_n, ds.min_n, ds.first_n, ds.last_n])
def test_compile_components_reuses_functions_for_n(reduction, antialias):
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "v": [1.0, 2.0]})
    _, info1, append1, *_, aa_funcs1, _ = _compile_components_for(
        df, reduction("v", n=2), antialias)
    _, info2, append2, *_, aa_funcs2, _ = _compile_components_for(
        df, reduction("v", n=3), antialias)
    assert info1 is info2
    assert append1 is append2
    if antialias:
        assert aa_funcs1 == aa_funcs2


def test_compile_components_reuses_functions_for_categories():
    df1 = pd.DataFrame({"x": [0.0], "y": [0.0], "cat": pd.Categorical(["a"])})
    df2 = pd.DataFrame({"x": [0.0], "y": [0.0], "cat": pd.Categorical(["b"])})
    _, info1, append1, *_ = _compile_components_for(df1, ds.by("cat", ds.count()))
    _, info2, append2, *_ = _compile_components_for(df2, ds.by("cat", ds.count()))
    assert info1 is info2
    assert append1 is append2


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("selector", [ds.max("v"), ds.max_n("v", n=2), ds.first("v")])
def test_where_combine_callback_is_reused(selector, categorical):
    combine1 = ds.where(selector, "a")._combine_callback(False, False, categorical)
    combine2 = ds.where(selector, "b")._combine_callback(False, False, categorical)
    assert combine1 is combine2


def test_extend_compiled_once_for_n():
    from numba.core import event

    class CompileNames(event.Listener):
        def __init__(self):
            super().__init__()
            self.names = []

        def on_start(self, ev):
            self.names.append(ev.data["dispatcher"].py_func.__name__)

        def on_end(self, ev):
            pass

    df = pd.DataFrame({
        "x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0],
        "v_compiled_once_for_n": [1.0, 2.0, 3.0],
    })
    cvs = ds.Canvas(plot_width=3, plot_height=3)
    listener = CompileNames()
    with event.install_listener("numba:compile", listener):
        for n in range(1, 5):
            cvs.points(df, "x", "y", ds.max_n("v_compiled_once_for_n", n=n))
    assert listener.names.count("extend_cpu") <= 1
