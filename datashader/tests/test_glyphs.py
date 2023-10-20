from __future__ import annotations
import pandas as pd
import numpy as np
import pytest

from datashader.datashape import dshape
from datashader.glyphs import Point, LinesAxis1, Glyph
from datashader.glyphs.area import _build_draw_trapezoid_y
from datashader.glyphs.line import (
    _build_map_onto_pixel_for_line,
    _build_draw_segment,
    _build_extend_line_axis0,
)
from datashader.glyphs.trimesh import(
    _build_map_onto_pixel_for_triangle,
    _build_draw_triangle,
    _build_extend_triangles
)
from datashader.utils import ngjit


def test_point_bounds_check():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [5, 6, 7]})
    p = Point('x', 'y')
    assert p._compute_bounds(df['x'].values) == (1, 3)
    assert p._compute_bounds(df['y'].values) == (5, 7)


def test_point_validate():
    p = Point('x', 'y')
    p.validate(dshape("{x: int32, y: float32}"))
    with pytest.raises(ValueError):
        p.validate(dshape("{x: string, y: float32}"))


@ngjit
def append(i, x, y, agg):
    agg[y, x] += 1

@ngjit
def tri_append(x, y, agg, n):
    agg[y, x] += n


def new_agg():
    return np.zeros((5, 5), dtype='i4')


mapper = ngjit(lambda x: x)
map_onto_pixel_for_line = _build_map_onto_pixel_for_line(mapper, mapper)
map_onto_pixel_for_triangle = _build_map_onto_pixel_for_triangle(mapper, mapper)

# Line rasterization
expand_aggs_and_cols = Glyph._expand_aggs_and_cols(append, 1, False)
_draw_segment = _build_draw_segment(append, map_onto_pixel_for_line,
                                    expand_aggs_and_cols, 0, False)
extend_line, _ = _build_extend_line_axis0(_draw_segment, expand_aggs_and_cols, None)

# Triangles rasterization
draw_triangle, draw_triangle_interp = _build_draw_triangle(tri_append)
extend_triangles = _build_extend_triangles(draw_triangle, draw_triangle_interp,
                                           map_onto_pixel_for_triangle)

# Trapezoid y rasterization
_draw_trapezoid = _build_draw_trapezoid_y(
    append, map_onto_pixel_for_line, expand_aggs_and_cols
)

bounds = (-3, 1, -3, 1)
vt = (1., 3., 1., 3.)


def draw_segment(x0, y0, x1, y1, i, segment_start, agg):
    """
    Helper to draw line with fixed bounds and scale values.
    """
    sx, tx, sy, ty = 1, 0, 1, 0
    xmin, xmax, ymin, ymax = 0, 5, 0, 5
    buffer = np.empty(0)
    _draw_segment(
        i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
        segment_start, False, x0, x1, y0, y1, 0.0, 0.0, buffer, agg)


def draw_trapezoid(x0, x1, y0, y1, y2, y3, i, trapezoid_start, stacked, agg):
    """
    Helper to draw line with fixed bounds and scale values.
    """
    sx, tx, sy, ty = 1, 0, 1, 0
    xmin, xmax, ymin, ymax = 0, 5, 0, 5
    _draw_trapezoid(
        i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
        x0, x1, y0, y1, y2, y3, trapezoid_start, stacked, agg)


def test_draw_line():
    x0, y0 = (0, 0)
    x1, y1 = (3, 3)
    out = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0]])
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, True, agg)
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_segment(x1, y1, x0, y0, 0, True, agg)
    np.testing.assert_equal(agg, out)
    # plot_start = False
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, False, agg)
    out[0, 0] = 0
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_segment(x1, y1, x0, y0, 0, False, agg)
    out[0, 0] = 1
    out[3, 3] = 0
    np.testing.assert_equal(agg, out)
    # Flip coords
    x0, y0 = (0, 4)
    x1, y1 = (3, 1)
    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0]])
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, True, agg)
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_segment(x1, y1, x0, y0, 0, True, agg)
    np.testing.assert_equal(agg, out)
    # plot_start = False
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, False, agg)
    out[4, 0] = 0
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_segment(x1, y1, x0, y0, 0, False, agg)
    out[4, 0] = 1
    out[1, 3] = 0


def test_draw_line_same_point():
    x0, y0 = (4, 4)
    x1, y1 = (4, 4)
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, True, agg)
    assert agg.sum() == 1
    assert agg[4, 4] == 1
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, False, agg)
    assert agg.sum() == 1
    assert agg[4, 4] == 1

    x0, y0 = (4, 4)
    x1, y1 = (10, 10)
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, True, agg)
    assert agg.sum() == 1
    assert agg[4, 4] == 1
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, False, agg)
    assert agg.sum() == 0
    assert agg[4, 4] == 0


def test_draw_line_vertical_horizontal():
    # Vertical
    x0, y0 = (3, 3)
    x1, y1 = (3, 0)
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, True, agg)
    out = new_agg()
    out[:4, 3] = 1
    np.testing.assert_equal(agg, out)
    # Horizontal
    agg = new_agg()
    draw_segment(y0, x0, y1, x1, 0, True, agg)
    out = new_agg()
    out[3, :4] = 1
    np.testing.assert_equal(agg, out)


def test_extend_lines():
    xs = np.array([0, -2, -2, 0, 0])
    ys = np.array([-1, -1, 1.1, 1.1, -1])
    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0]])
    agg = new_agg()
    sx, tx, sy, ty = vt
    xmin, xmax, ymin, ymax = bounds
    buffer = np.empty(0)
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, False, buffer, agg)
    np.testing.assert_equal(agg, out)
    # plot_start = True
    out[2, 3] += 1
    agg = new_agg()
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, True, buffer, agg)
    np.testing.assert_equal(agg, out)

    xs = np.array([2, 1, 0, -1, -4, -1, -100, -1, 2])
    ys = np.array([-1, -2, -3, -4, -1, 2, 100, 2, -1])
    out = np.array([[0, 1, 0, 1, 0],
                    [1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0]])
    agg = new_agg()
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, True, buffer, agg)
    np.testing.assert_equal(agg, out)


def test_extend_lines_all_out_of_bounds():
    xs = np.array([-100, -200, -100])
    ys = np.array([0, 0, 1])
    agg = new_agg()
    sx, tx, sy, ty = vt
    xmin, xmax, ymin, ymax = bounds
    buffer = np.empty(0)
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, True, buffer, agg)
    assert agg.sum() == 0


def test_extend_lines_nan():
    xs = np.array([-3, -2, np.nan, 0, 1])
    ys = np.array([-3, -2, np.nan, 0, 1])
    agg = new_agg()
    sx, tx, sy, ty = vt
    xmin, xmax, ymin, ymax = bounds
    buffer = np.empty(0)
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, True, buffer, agg)
    out = np.diag([1, 1, 0, 1, 0])
    np.testing.assert_equal(agg, out)


def test_extend_lines_exact_bounds():
    xs = np.array([-3, 1, 1, -3, -3])
    ys = np.array([-3, -3, 1, 1, -3])

    agg = np.zeros((4, 4), dtype='i4')
    sx, tx, sy, ty = vt
    xmin, xmax, ymin, ymax = bounds
    buffer = np.empty(0)
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, True, buffer, agg)
    out = np.array([[2, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1]])
    np.testing.assert_equal(agg, out)

    agg = np.zeros((4, 4), dtype='i4')
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, False, buffer, agg)
    out = np.array([[1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1]])
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_acute():
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (1, 3, 4, 0)

    out = np.array([[0, 0, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from left to right
    trapezoid_start = True
    stacked = True
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_acute_not_stacked():
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (1, 3, 4, 0)

    out = np.array([[0, 0, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0]])

    # Specify vertices from left to right
    trapezoid_start = True
    stacked = False
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_right():
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (1, 3, 4, 1)

    out = np.array([[0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from left to right
    trapezoid_start = True
    stacked = True
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_obtuse():
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (0, 3, 5, 1)

    out = np.array([[1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from left to right
    trapezoid_start = True
    stacked = True
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_intersecting():
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (0, 5, 1, 4)

    out = np.array([[1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 0],
                    [1, 0, 1, 1, 0],
                    [0, 0, 0, 1, 0]])

    # Specify vertices from left to right
    trapezoid_start = True
    stacked = True
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_vertical_line_start_and_not_clipped():
    x0, x1 = (2, 2)
    y0, y1, y2, y3 = (1, 3, 4, 0)

    out = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from inner to outer
    trapezoid_start = True
    stacked = True
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from outer to inner which should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_vertical_line_not_start_and_not_clipped():
    x0, x1 = (2, 2)
    y0, y1, y2, y3 = (1, 3, 4, 0)
    trapezoid_start = False
    stacked = True

    # trapezoid_start=False, clipped=False
    out = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from inner to outer
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from outer to inner
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_clipped():
    x0, x1 = (4, 6)
    y0, y1, y2, y3 = (1, 3, 5, 0)
    trapezoid_start = True
    stacked = True

    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from inner to outer
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    out = np.array([[0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from outer to inner
    agg = new_agg()
    draw_trapezoid(x1, x0, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_vertical_line_not_start_and_clipped():
    x0, x1 = (4, 6)
    y0, y1, y2, y3 = (1, 3, 4, 0)
    trapezoid_start = False
    stacked = True

    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from inner to outer
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_horizontal_line():
    # Obtuse trapezoid
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (2, 2, 2, 2)
    trapezoid_start = True
    stacked = False

    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from left to right
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    # with stacked = True, the zero width line is not rendered
    stacked = True
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg.sum(), 0)


def test_draw_trapezoid_diagonal_line():
    # Obtuse trapezoid
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (0, 0, 2, 2)
    trapezoid_start = True
    stacked = False

    out = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from left to right
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    # with stacked = True, the zero width line is not rendered
    stacked = True
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg.sum(), 0)


def test_draw_trapezoid_point():
    # Obtuse trapezoid
    x0, x1 = (3, 3)
    y0, y1, y2, y3 = (2, 2, 2, 2)
    trapezoid_start = True
    stacked = False

    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from left to right
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    # trapezoid_start=False and clipped=False causes only a single aggregation in
    # the point bin
    trapezoid_start = False
    out[2, 3] = 1
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)

    # with stacked = True, the zero width line is not rendered
    trapezoid_start = True
    stacked = True
    out[2, 3] = 0
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0,
                   trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_triangle_nointerp():
    """Assert that we draw triangles properly, without interpolation enabled.
    """
    # Isosceles triangle
    tri = ((2, -0.5), (-0.5, 2.5), (4.5, 2.5))
    out = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (0, 4, 0, 3), (0, 0, 0), (agg,), 1)
    np.testing.assert_equal(agg, out)

    # Right triangle
    tri = ((2.4, -0.5), (-0.5, 2.4), (2.4, 2.4))
    out = np.array([[0, 0, 2, 0, 0],
                    [0, 2, 2, 0, 0],
                    [2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (0, 4, 0, 3), (0, 0, 0), (agg,), 2)
    np.testing.assert_equal(agg, out)

    # Two right trimesh
    tri = ((2.4, -0.5), (-0.5, 2.4), (2.4, 2.4),
           (2.4, -0.5), (2.4, 3.5), (4.5, -0.5))
    out = np.array([[0, 0, 3, 4, 4],
                    [0, 3, 3, 4, 0],
                    [3, 3, 3, 4, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri[:3], (0, 4, 0, 3), (0, 0, 0), (agg,), 3)
    draw_triangle(tri[3:], (0, 4, 0, 3), (0, 0, 0), (agg,), 4)
    np.testing.assert_equal(agg, out)

    # Draw isoc triangle with clipping
    tri = ((2, -0.5), (-0.5, 2.5), (4.5, 2.5))
    out = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (0, 3, 0, 2), (0, 0, 0), (agg,), 1)
    np.testing.assert_equal(agg, out)

    # clip from right and left
    out = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (1, 3, 0, 2), (0, 0, 0), (agg,), 1)
    np.testing.assert_equal(agg, out)

    # clip from right, left, top
    out = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (1, 3, 1, 2), (0, 0, 0), (agg,), 1)
    np.testing.assert_equal(agg, out)

    # clip from right, left, top, bottom
    out = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (1, 3, 1, 1), (0, 0, 0), (agg,), 1)
    np.testing.assert_equal(agg, out)

def test_draw_triangle_interp():
    """Assert that we draw triangles properly, with interpolation enabled.
    """
    # Isosceles triangle
    tri = ((2, -0.5), (-0.5, 2.5), (4.5, 2.5))
    out = np.array([[0, 0, 3, 0, 0],
                    [0, 3, 3, 3, 0],
                    [3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle_interp(tri, (0, 4, 0, 3), (0, 0, 0), (agg,), (3, 3, 3))
    np.testing.assert_equal(agg, out)

    tri = ((2, -0.5), (-0.5, 2.5), (4.5, 2.5))
    out = np.array([[0,    0,    1.25, 0,    0   ],
                    [0,    1.55, 1.75, 1.95, 0   ],
                    [1.85, 2.05, 2.25, 2.45, 2.65],
                    [0,    0,    0, 0, 0]])
    agg = np.zeros((4, 5), dtype='f4')
    draw_triangle_interp(tri, (0, 4, 0, 2), (0, 0, 0), (agg,), (1, 2, 3))
    np.testing.assert_allclose(agg, out)

    tri = ((2, -0.5), (-0.5, 2.5), (4.5, 2.5))
    out = np.array([[0,    0,    3.75, 0,    0   ],
                    [0,    4.65, 5.25, 5.85, 0   ],
                    [5.55, 6.15, 6.75, 7.35, 7.95],
                    [0,    0,    0,    0,    0   ]])
    agg = np.zeros((4, 5), dtype='f4')
    draw_triangle_interp(tri, (0, 4, 0, 2), (0, 0, 0), (agg,), (3, 6, 9))
    np.testing.assert_allclose(agg, out)

    tri = ((2, -0.5), (-0.5, 2.5), (4.5, 2.5))
    out = np.array([[0,   0,   5.5, 0,   0  ],
                    [0,   4.9, 4.5, 4.1, 0  ],
                    [4.3, 3.9, 3.5, 3.1, 2.7],
                    [0,   0,   0,   0,   0  ]])
    agg = np.zeros((4, 5), dtype='f4')
    draw_triangle_interp(tri, (0, 4, 0, 2), (0, 0, 0), (agg,), (6, 4, 2))
    np.testing.assert_allclose(agg, out)

def test_draw_triangle_subpixel():
    """Assert that we draw subpixel triangles properly, both with and without
    interpolation.
    """
    # With interpolation
    tri = ((2, -0.5), (-0.5, 2.5), (4.5, 2.5),
           (2, 3), (2, 3), (2, 3),
           (2, 3), (2, 3), (2, 3))
    out = np.array([[0,   0,   5.5, 0,   0  ],
                    [0,   4.9, 4.5, 4.1, 0  ],
                    [4.3, 3.9, 3.5, 3.1, 2.7],
                    [0,   0,   8,   0,   0  ]])
    agg = np.zeros((4, 5), dtype='f4')
    draw_triangle_interp(tri[:3], (0, 4, 0, 5), (0, 0, 0), (agg,), (6, 4, 2))
    draw_triangle_interp(tri[3:6], (2, 2, 3, 3), (0, 0, 0), (agg,), (6, 4, 2))
    draw_triangle_interp(tri[6:], (2, 2, 3, 3), (0, 0, 0), (agg,), (6, 4, 2))
    np.testing.assert_allclose(agg, out)

    # Without interpolation
    out = np.array([[0, 0, 2, 0, 0],
                    [0, 2, 2, 2, 0],
                    [2, 2, 2, 2, 2],
                    [0, 0, 4, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri[:3], (0, 4, 0, 5), (0, 0, 0), (agg,), 2)
    draw_triangle(tri[3:6], (2, 2, 3, 3), (0, 0, 0), (agg,), 2)
    draw_triangle(tri[6:], (2, 2, 3, 3), (0, 0, 0), (agg,), 2)
    np.testing.assert_equal(agg, out)


def test_line_awkward_point_on_upper_bound_maps_to_last_pixel():
    """Check that point deliberately chosen to be on the upper bound but
    with a similar-magnitudes subtraction error like that which could
    occur in extend line does indeed get mapped to last pixel.
    """
    num_y_pixels = 2
    ymax = 0.1
    bigy = 10e9

    sy = num_y_pixels/ymax
    y = bigy-(bigy-ymax) # simulates clipped line

    # check that test is set up ok
    assert y!=ymax
    np.testing.assert_almost_equal(y,ymax,decimal=6)

    _,pymax = map_onto_pixel_for_line(1.0, 0.0, sy, 0.0,
                                      0.0, 1.0, 0.0, ymax,
                                      1.0, y)

    assert pymax==num_y_pixels-1


def test_lines_xy_validate():
    g = LinesAxis1(['x0', 'x1'], ['y11', 'y12'])
    g.validate(
        dshape("{x0: int32, x1: int32, y11: float32, y12: float32}"))

    with pytest.raises(ValueError):
        g.validate(
            dshape("{x0: int32, x1: float32, y11: string, y12: float32}"))
