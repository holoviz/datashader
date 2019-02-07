import pytest

import numpy as np

from datashader.glyphs import _build_draw_line, _build_extend_line_axis0, _build_map_onto_pixel_for_line
from datashader.utils import ngjit


@pytest.fixture
def extend_line():
    @ngjit
    def append(i, x, y, agg):
        agg[y, x] += 1

    mapper = ngjit(lambda x: x)
    map_onto_pixel = _build_map_onto_pixel_for_line(mapper, mapper)
    draw_line = _build_draw_line(append)
    return _build_extend_line_axis0(draw_line, map_onto_pixel)


@pytest.mark.parametrize('high', [0, 10**5])
@pytest.mark.parametrize('low', [0, -10**5])
@pytest.mark.benchmark(group="extend_line")
def test_extend_line_uniform(benchmark, extend_line, low, high):
    n = 10**6
    vt = (1, 0, 1, 0)
    bounds = (0, 0, 10**4, 10**4)

    xs = np.random.uniform(bounds[0] + low, bounds[2] + high, n)
    ys = np.random.uniform(bounds[1] + low, bounds[3] + high, n)

    agg = np.zeros((bounds[2], bounds[3]), dtype='i4')
    benchmark(extend_line, vt, bounds, xs, ys, True, agg)


@pytest.mark.benchmark(group="extend_line")
def test_extend_line_normal(benchmark, extend_line):
    n = 10**6
    vt = (1, 0, 1, 0)
    bounds = (0, 0, 10**4, 10**4)

    start = 1456297053
    end = start + 60 * 60 * 24
    xs = np.linspace(start, end, n)

    signal = np.random.normal(0, 0.3, size=n).cumsum() + 50
    noise = lambda var, bias, n: np.random.normal(bias, var, n)
    ys = signal + noise(1, 10*(np.random.random() - 0.5), n)

    agg = np.zeros((bounds[2], bounds[3]), dtype='i4')
    benchmark(extend_line, vt, bounds, xs, ys, True, agg)
