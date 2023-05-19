from __future__ import annotations

import pytest

import numpy as np

from datashader.glyphs import Glyph
from datashader.glyphs.line import _build_draw_segment, \
    _build_map_onto_pixel_for_line
from datashader.utils import ngjit

mapper = ngjit(lambda x: x)
map_onto_pixel = _build_map_onto_pixel_for_line(mapper, mapper)
sx, tx, sy, ty = 1, 0, 1, 0
xmin, xmax, ymin, ymax = 0, 5, 0, 5


@pytest.fixture
def draw_line():
    @ngjit
    def append(i, x, y, agg):
        agg[y, x] += 1

    expand_aggs_and_cols = Glyph._expand_aggs_and_cols(append, 1, False)
    return _build_draw_segment(append, map_onto_pixel, expand_aggs_and_cols,
                               0, False)


@pytest.mark.benchmark(group="draw_line")
def test_draw_line_left_border(benchmark, draw_line):
    n = 10**4
    x0, y0 = (0, 0)
    x1, y1 = (0, n)

    agg = np.zeros((n+1, n+1), dtype='i4')
    buffer = np.empty(0)
    benchmark(draw_line, 0, sx, tx, sy, ty, xmin, xmax, ymin, ymax, True, True,
              x0, x1, y0, y1, 0.0, 0.0, buffer, agg)


@pytest.mark.benchmark(group="draw_line")
def test_draw_line_diagonal(benchmark, draw_line):
    n = 10**4
    x0, y0 = (0, 0)
    x1, y1 = (n, n)

    agg = np.zeros((n+1, n+1), dtype='i4')
    buffer = np.empty(0)
    benchmark(draw_line, 0, sx, tx, sy, ty, xmin, xmax, ymin, ymax, True, True,
              x0, x1, y0, y1, 0.0, 0.0, buffer, agg)


@pytest.mark.benchmark(group="draw_line")
def test_draw_line_offset(benchmark, draw_line):
    n = 10**4
    x0, y0 = (0, n//4)
    x1, y1 = (n, n//4-1)

    agg = np.zeros((n+1, n+1), dtype='i4')
    buffer = np.empty(0)
    benchmark(draw_line, 0, sx, tx, sy, ty, xmin, xmax, ymin, ymax, True, True,
              x0, x1, y0, y1, 0.0, 0.0, buffer, agg)
