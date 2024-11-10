import pytest
import pandas as pd
import datashader as ds


@pytest.fixture
def time_series(rng):
    n = 10**7
    signal = rng.normal(0, 0.3, size=n).cumsum() + 50
    def noise(var, bias, n):
        return rng.normal(bias, var, n)
    ys = signal + noise(1, 10*(rng.random() - 0.5), n)

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


@pytest.mark.gpu
@pytest.mark.benchmark(group="canvas")
def test_line_gpu(benchmark, time_series):
    from cudf import from_pandas
    time_series = from_pandas(time_series)
    cvs = ds.Canvas(plot_height=300, plot_width=900)
    benchmark(cvs.line, time_series, 'x', 'y')


@pytest.mark.gpu
@pytest.mark.benchmark(group="canvas")
def test_points_gpu(benchmark, time_series):
    from cudf import from_pandas
    time_series = from_pandas(time_series)
    cvs = ds.Canvas(plot_height=300, plot_width=900)
    benchmark(cvs.points, time_series, 'x', 'y')
