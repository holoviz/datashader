import pandas as pd
import pytest

import datashader as ds
import datashader.transfer_functions as tf


@pytest.fixture
def categorical_data(rng):
    def _gen(n_points, n_categories):
        xy = rng.standard_normal((n_points, 2))
        categories = [chr(65 + i) for i in range(n_categories)]
        c = rng.choice(categories, size=n_points)
        df = pd.DataFrame(xy, columns=["x", "y"])
        df["c"] = pd.Categorical(c, categories=categories)
        return df
    return _gen


@pytest.fixture
def categorical_agg(categorical_data):
    def _agg(n_points, n_categories, size):
        df = categorical_data(n_points, n_categories)
        cvs = ds.Canvas(plot_width=size, plot_height=size)
        return cvs.points(df, x="x", y="y", agg=ds.count_cat("c"))
    return _agg


@pytest.mark.benchmark(group="shade")
@pytest.mark.parametrize("n_categories", [5, 20])
@pytest.mark.parametrize("size", [10, 100, 1000])
def test_shade_categorical(benchmark, categorical_agg, n_categories, size):
    n_points = 10_000_000
    agg = categorical_agg(n_points, n_categories, size)
    benchmark(tf.shade, agg)


@pytest.fixture
def numeric_agg(rng):
    def _agg(n_points, size):
        xy = rng.standard_normal((n_points, 2))
        df = pd.DataFrame(xy, columns=["x", "y"])
        cvs = ds.Canvas(plot_width=size, plot_height=size)
        return cvs.points(df, x="x", y="y", agg=ds.count())
    return _agg


@pytest.mark.benchmark(group="shade")
@pytest.mark.parametrize("size", [10, 100, 1000])
def test_shade_numeric(benchmark, numeric_agg, size):
    """Benchmark tf.shade on numeric (non-categorical) aggregations."""
    n_points = 10_000_000
    agg = numeric_agg(n_points, size)
    benchmark(tf.shade, agg)
