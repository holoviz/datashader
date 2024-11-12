import numpy as np
import pytest

from datashader.glyphs import Glyph
from datashader.glyphs.line import (
    _build_draw_segment, _build_extend_line_axis0, _build_map_onto_pixel_for_line
)
from datashader.utils import ngjit


@pytest.fixture
def extend_line():
    @ngjit
    def append(i, x, y, agg):
        agg[y, x] += 1

    mapper = ngjit(lambda x: x)
    map_onto_pixel = _build_map_onto_pixel_for_line(mapper, mapper)
    expand_aggs_and_cols = Glyph._expand_aggs_and_cols(append, 1, False)
    draw_line = _build_draw_segment(append, map_onto_pixel,
                                    expand_aggs_and_cols, 0, False)
    return _build_extend_line_axis0(draw_line, expand_aggs_and_cols, None)[0]


@pytest.mark.parametrize('high', [0, 10**5])
@pytest.mark.parametrize('low', [0, -10**5])
@pytest.mark.benchmark(group="extend_line")
def test_extend_line_uniform(benchmark, extend_line, low, high, rng):
    n = 10**6
    sx, tx, sy, ty = (1, 0, 1, 0)
    xmin, xmax, ymin, ymax = (0, 0, 10**4, 10**4)

    xs = rng.uniform(xmin + low, ymin + high, n)
    ys = rng.uniform(xmax + low, ymax + high, n)

    agg = np.zeros((ymin, ymax), dtype='i4')
    buffer = np.empty(0)
    benchmark(
        extend_line, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, True, buffer, agg
    )


@pytest.mark.benchmark(group="extend_line")
def test_extend_line_normal(benchmark, extend_line, rng):
    n = 10**6
    sx, tx, sy, ty = (1, 0, 1, 0)
    xmin, xmax, ymin, ymax = (0, 0, 10**4, 10**4)

    start = 1456297053
    end = start + 60 * 60 * 24
    xs = np.linspace(start, end, n)

    signal = rng.normal(0, 0.3, size=n).cumsum() + 50
    def noise(var, bias, n):
        return rng.normal(bias, var, n)
    ys = signal + noise(1, 10*(rng.random() - 0.5), n)

    agg = np.zeros((ymin, ymax), dtype='i4')
    buffer = np.empty(0)
    benchmark(
        extend_line, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, True, buffer, agg
    )
