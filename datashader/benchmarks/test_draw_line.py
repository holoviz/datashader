from __future__ import division

import pytest

import numpy as np

from datashader.glyphs import _build_draw_line
from datashader.utils import ngjit


@pytest.fixture
def draw_line():
    @ngjit
    def append(i, x, y, agg):
        agg[y, x] += 1

    return _build_draw_line(append)


@pytest.mark.benchmark(group="draw_line")
def test_draw_line_left_border(benchmark, draw_line):
    n = 10**4
    x0, y0 = (0, 0)
    x1, y1 = (0, n)

    agg = np.zeros((n+1, n+1), dtype='i4')
    benchmark(draw_line, x0, y0, x1, y1, 0, True, False, agg)


@pytest.mark.benchmark(group="draw_line")
def test_draw_line_diagonal(benchmark, draw_line):
    n = 10**4
    x0, y0 = (0, 0)
    x1, y1 = (n, n)

    agg = np.zeros((n+1, n+1), dtype='i4')
    benchmark(draw_line, x0, y0, x1, y1, 0, True, False, agg)


@pytest.mark.benchmark(group="draw_line")
def test_draw_line_offset(benchmark, draw_line):
    n = 10**4
    x0, y0 = (0, n//4)
    x1, y1 = (n, n//4-1)

    agg = np.zeros((n+1, n+1), dtype='i4')
    benchmark(draw_line, x0, y0, x1, y1, 0, True, False, agg)
