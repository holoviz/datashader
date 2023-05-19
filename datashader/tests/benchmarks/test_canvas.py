import pytest
import os
import numpy as np
import pandas as pd

import datashader as ds

test_gpu = bool(int(os.getenv("DATASHADER_TEST_GPU", 0)))


@pytest.fixture
def time_series():
    n = 10**7
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


@pytest.mark.skipif(not test_gpu, reason="DATASHADER_TEST_GPU not set")
@pytest.mark.benchmark(group="canvas")
def test_line_gpu(benchmark, time_series):
    from cudf import from_pandas
    time_series = from_pandas(time_series)
    cvs = ds.Canvas(plot_height=300, plot_width=900)
    benchmark(cvs.line, time_series, 'x', 'y')


@pytest.mark.skipif(not test_gpu, reason="DATASHADER_TEST_GPU not set")
@pytest.mark.benchmark(group="canvas")
def test_points_gpu(benchmark, time_series):
    from cudf import from_pandas
    time_series = from_pandas(time_series)
    cvs = ds.Canvas(plot_height=300, plot_width=900)
    benchmark(cvs.points, time_series, 'x', 'y')
