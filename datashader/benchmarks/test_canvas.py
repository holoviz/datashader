import pytest

import numpy as np
import pandas as pd

import datashader as ds


@pytest.fixture
def time_series():
    n = 10**6
    signal = np.random.normal(0, 0.3, size=n).cumsum() + 50
    noise = lambda var, bias, n: np.random.normal(bias, var, n)
    ys = signal + noise(1, 10*(np.random.random() - 0.5), n)

    df = pd.DataFrame({'y': ys})
    df['x'] = df.index
    return df


@pytest.mark.benchmark(group="canvas")
def test_line(benchmark, time_series):
    cvs = ds.Canvas(plot_height=300, plot_width=900)
    benchmark(cvs.line, time_series, 'x', 'y')


@pytest.mark.benchmark(group="canvas")
def test_points(benchmark, time_series):
    cvs = ds.Canvas(plot_height=300, plot_width=900)
    benchmark(cvs.points, time_series, 'x', 'y')
