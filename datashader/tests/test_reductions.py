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


@pytest.mark.parametrize("a, b", [
    (
        ds.reductions.category_values(ds.category_modulo('c', 3), 'v'),
        ds.reductions.category_values(ds.category_binning('c', 0, 2, 3), 'v'),
    ),
    (ds.by(ds.category_modulo('c', 3), ds.max('v')), ds.by(ds.category_modulo('c', 4), ds.max('v'))),
    (ds.by('c', ds.max('v')), ds.by('c', ds.min('v'))),
    (ds.where(ds.max('v'), 'w'), ds.where(ds.min('v'), 'w')),
])
def test_hash_and_eq_include_nested_expressions(a, b):
    assert a != b
    assert hash(a) != hash(b)
    assert len({a, b}) == 2


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_compile_components_reused_across_unused_columns(dtype):
    from datashader.compiler import compile_components
    from datashader.glyphs import LineAxis0
    from datashader.utils import dshape_from_pandas

    df = pd.DataFrame({'x': [0.0, 1.0], 'y': [0.0, 1.0], 'v': [1.0, 2.0]})
    other = df.assign(unused=np.zeros(2, dtype=dtype))
    glyph = LineAxis0('x', 'y')
    glyph.set_line_width(1)
    agg = ds.summary(max=ds.max('v'), first=ds.first('v'))

    def compile(df):
        return compile_components(agg, dshape_from_pandas(df), glyph, antialias=True)

    _, info1, append1, *_, aa_funcs1, _ = compile(df)
    _, info2, append2, *_, aa_funcs2, _ = compile(other)
    assert append1 is append2
    assert info1 is info2
    assert aa_funcs1 is aa_funcs2
