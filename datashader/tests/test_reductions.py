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
