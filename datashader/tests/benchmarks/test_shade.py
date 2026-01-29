import pandas as pd
import pytest

import datashader as ds
import datashader.transfer_functions as tf


@pytest.mark.benchmark(group="shade")
@pytest.mark.parametrize("n_categories", [1, 5, 20])
@pytest.mark.parametrize("size", [10, 100, 1000])
def test_shade(benchmark, rng, size, n_categories):
    N = 1_000_000
    categories = [chr(65 + i) for i in range(n_categories)]
    df = pd.DataFrame(
        {
            "x": rng.standard_normal(N),
            "y": rng.standard_normal(N),
            "c": pd.Categorical(rng.choice(categories, size=N), categories=categories),
        }
    )

    cvs = ds.Canvas(plot_width=size, plot_height=size)
    agg = cvs.points(df, x="x", y="y", agg=ds.count_cat("c"))

    benchmark(tf.shade, agg)
