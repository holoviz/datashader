from __future__ import absolute_import
from datashape import dshape
import pandas as pd
import numpy as np
import pytest

from datashader.glyphs import Point, LinesAxis1, Glyph

from datashader.glyphs.area import _build_draw_trapezoid_y
from datashader.glyphs.line import (
    _build_map_onto_pixel_for_line,
    _build_draw_line,
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
    assert p._compute_x_bounds(df['x'].values) == (1, 3)
    assert p._compute_y_bounds(df['y'].values) == (5, 7)


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
expand_aggs_and_cols = Glyph._expand_aggs_and_cols(append, 1)
draw_line = _build_draw_line(append, expand_aggs_and_cols)
extend_line = _build_extend_line_axis0(
    draw_line, map_onto_pixel_for_line, expand_aggs_and_cols
)

# Triangles rasterization
draw_triangle, draw_triangle_interp = _build_draw_triangle(tri_append)
extend_triangles = _build_extend_triangles(draw_triangle, draw_triangle_interp, map_onto_pixel_for_triangle)

# Trapezoid y rasterization
draw_trapezoid = _build_draw_trapezoid_y(append, expand_aggs_and_cols)

bounds = (-3, 1, -3, 1)
vt = (1., 3., 1., 3.)

def test_draw_line():
    x0, y0 = (0, 0)
    x1, y1 = (3, 3)
    out = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0]])
    agg = new_agg()
    draw_line(x0, y0, x1, y1, 0, True, False, agg)
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_line(x1, y1, x0, y0, 0, True, False, agg)
    np.testing.assert_equal(agg, out)
    # plot_start = False
    agg = new_agg()
    draw_line(x0, y0, x1, y1, 0, False, False, agg)
    out[0, 0] = 0
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_line(x1, y1, x0, y0, 0, False, False, agg)
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
    draw_line(x0, y0, x1, y1, 0, True, False, agg)
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_line(x1, y1, x0, y0, 0, True, False, agg)
    np.testing.assert_equal(agg, out)
    # plot_start = False
    agg = new_agg()
    draw_line(x0, y0, x1, y1, 0, False, False, agg)
    out[4, 0] = 0
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_line(x1, y1, x0, y0, 0, False, False, agg)
    out[4, 0] = 1
    out[1, 3] = 0


def test_draw_line_same_point():
    x0, y0 = (3, 3)
    x1, y1 = (3, 3)
    agg = new_agg()
    draw_line(x0, y0, x1, y1, 0, True, False, agg)
    assert agg.sum() == 2
    assert agg[3, 3] == 2
    agg = new_agg()
    draw_line(x0, y0, x1, y1, 0, False, False, agg)
    assert agg.sum() == 1
    assert agg[3, 3] == 1
    agg = new_agg()
    draw_line(x0, y0, x1, y1, 0, True, True, agg)
    assert agg.sum() == 1
    assert agg[3, 3] == 1
    agg = new_agg()
    draw_line(x0, y0, x1, y1, 0, False, True, agg)
    assert agg.sum() == 0
    assert agg[3, 3] == 0


def test_draw_line_vertical_horizontal():
    # Vertical
    x0, y0 = (3, 3)
    x1, y1 = (3, 0)
    agg = new_agg()
    draw_line(x0, y0, x1, y1, 0, True, False, agg)
    out = new_agg()
    out[:4, 3] = 1
    np.testing.assert_equal(agg, out)
    # Horizontal
    agg = new_agg()
    draw_line(y0, x0, y1, x1, 0, True, False, agg)
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
    extend_line(vt, bounds, xs, ys, False, agg)
    np.testing.assert_equal(agg, out)
    # plot_start = True
    out[2, 3] += 1
    agg = new_agg()
    extend_line(vt, bounds, xs, ys, True, agg)
    np.testing.assert_equal(agg, out)

    xs = np.array([2, 1, 0, -1, -4, -1, -100, -1, 2])
    ys = np.array([-1, -2, -3, -4, -1, 2, 100, 2, -1])
    out = np.array([[0, 1, 0, 1, 0],
                    [1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0]])
    agg = new_agg()
    extend_line(vt, bounds, xs, ys, True, agg)
    np.testing.assert_equal(agg, out)


def test_extend_lines_all_out_of_bounds():
    xs = np.array([-100, -200, -100])
    ys = np.array([0, 0, 1])
    agg = new_agg()
    extend_line(vt, bounds, xs, ys, True, agg)
    assert agg.sum() == 0


def test_extend_lines_nan():
    xs = np.array([-3, -2, np.nan, 0, 1])
    ys = np.array([-3, -2, np.nan, 0, 1])
    agg = new_agg()
    extend_line(vt, bounds, xs, ys, True, agg)
    out = np.diag([1, 1, 0, 2, 0])
    np.testing.assert_equal(agg, out)


def test_extend_lines_exact_bounds():
    xs = np.array([-3, 1, 1, -3, -3])
    ys = np.array([-3, -3, 1, 1, -3])

    agg = np.zeros((4, 4), dtype='i4')
    extend_line(vt, bounds, xs, ys, True, agg)
    out = np.array([[2, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1]])
    np.testing.assert_equal(agg, out)

    agg = np.zeros((4, 4), dtype='i4')
    extend_line(vt, bounds, xs, ys, False, agg)
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
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Specify vertices from left to right
    plot_start = True
    clipped = False
    stacked = True
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_acute_not_stacked():
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (1, 3, 4, 0)

    out = np.array([[0, 0, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0]])
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Specify vertices from left to right
    plot_start = True
    clipped = False
    stacked = False
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_right():
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (1, 3, 4, 1)

    out = np.array([[0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0]])
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Specify vertices from left to right
    plot_start = True
    clipped = False
    stacked = True
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_obtuse():
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (0, 3, 5, 1)

    out = np.array([[1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0]])
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Specify vertices from left to right
    plot_start = True
    clipped = False
    stacked = True
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_intersecting():
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (0, 5, 1, 4)

    out = np.array([[1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 0, 0, 1, 0]])
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Specify vertices from left to right
    plot_start = True
    clipped = False
    stacked = True
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_vertical_line_start_and_not_clipped():
    x0, x1 = (2, 2)
    y0, y1, y2, y3 = (1, 3, 4, 0)

    out = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]])
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Specify vertices from inner to outer
    plot_start = True
    clipped = False
    stacked = True
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from outer to inner which should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_vertical_line_not_start_and_not_clipped():
    x0, x1 = (2, 2)
    y0, y1, y2, y3 = (1, 3, 4, 0)
    plot_start = False
    clipped = False
    stacked = True

    # plot_start=False, clipped=False
    out = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]])
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Specify vertices from inner to outer
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from outer to inner
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_vertical_line_start_and_clipped():
    x0, x1 = (2, 2)
    y0, y1, y2, y3 = (1, 3, 4, 0)
    plot_start = True
    clipped = True
    stacked = True

    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Specify vertices from inner to outer
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    out = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]])

    # Specify vertices from outer to inner
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_vertical_line_not_start_and_clipped():
    x0, x1 = (2, 2)
    y0, y1, y2, y3 = (1, 3, 4, 0)
    plot_start = False
    clipped = True
    stacked = True

    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Specify vertices from inner to outer
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from outer to inner
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_trapezoid_horizontal_line():
    # Obtuse trapezoid
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (2, 2, 2, 2)
    plot_start = True
    clipped = False
    stacked = False

    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Specify vertices from left to right
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # with stacked = True, the zero width line is not rendered
    stacked = True
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg.sum(), 0)


def test_draw_trapezoid_diagonal_line():
    # Obtuse trapezoid
    x0, x1 = (0, 3)
    y0, y1, y2, y3 = (0, 0, 2, 2)
    plot_start = True
    clipped = False
    stacked = False

    out = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Specify vertices from left to right
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # with stacked = True, the zero width line is not rendered
    stacked = True
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg.sum(), 0)


def test_draw_trapezoid_point():
    # Obtuse trapezoid
    x0, x1 = (3, 3)
    y0, y1, y2, y3 = (2, 2, 2, 2)
    plot_start = True
    clipped = False
    stacked = False

    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Specify vertices from left to right
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # plot_start=False and clipped=False causes only a single aggregation in
    # the point bin
    plot_start = False
    clipped = False
    out[2, 3] = 1
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # plot_start=True and clipt=True causes only a single aggregation in
    # the point bin
    plot_start = True
    clipped = True
    out[2, 3] = 1
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # plot_start=False and clipped=True causes no aggregation to be performed
    plot_start = False
    clipped = True
    out[2, 3] = 0
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out);

    # with stacked = True, the zero width line is not rendered
    plot_start = True
    clipped = False
    stacked = True
    out[2, 3] = 0
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out);


@pytest.mark.parametrize('shifty', range(-5, 6))
@pytest.mark.parametrize('shiftx', range(-5, 6))
def test_draw_trapezoid_with_clipping(shiftx, shifty):

    x0, x1 = (0 + shiftx, 3 + shiftx)
    y0, y1, y2, y3 = (1 + shifty, 3 + shifty, 4 + shifty, 0 + shifty)
    plot_start = True
    clipped = False
    stacked = True

    out = np.array([[0, 0, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0]])
    ymaxi, xmaxi = out.shape[0] - 1, out.shape[1] - 1

    # Shift expected output

    # shift x
    out = np.roll(out, shiftx, axis=1)
    if shiftx < 0:
        out[:, shiftx:] = 0
    else:
        out[:, :shiftx] = 0

    # shift y
    out = np.roll(out, shifty, axis=0)
    if shifty < 0:
        out[shifty:, :] = 0
    else:
        out[:shifty, :] = 0

    # Specify vertices from left to right
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)

    # Specify vertices from right to left should give same result
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, xmaxi, ymaxi, 0,
                   plot_start, clipped, stacked, agg)
    np.testing.assert_equal(agg, out)


def test_draw_triangle_nointerp():
    """Assert that we draw triangles properly, without interpolation enabled.
    """
    # Isosceles triangle
    tri = ((2, 0), (0, 2), (4, 2))
    out = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (0, 4, 0, 5), (0, 0, 0), (agg,), 1)
    np.testing.assert_equal(agg, out)

    # Right triangle
    tri = ((2, 0), (0, 2), (2, 2))
    out = np.array([[0, 0, 2, 0, 0],
                    [0, 2, 2, 0, 0],
                    [2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (0, 4, 0, 5), (0, 0, 0), (agg,), 2)
    np.testing.assert_equal(agg, out)

    # Two right trimesh
    tri = ((2, 0), (1, 1), (2, 1),
           (2, 1), (2, 2), (3, 2))
    out = np.array([[0, 0, 3, 0, 0],
                    [0, 3, 6, 0, 0],
                    [0, 0, 3, 3, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri[:3], (0, 4, 0, 5), (0, 0, 0), (agg,), 3)
    draw_triangle(tri[3:], (0, 4, 0, 5), (0, 0, 0), (agg,), 3)
    np.testing.assert_equal(agg, out)

    # Draw isoc triangle with clipping
    tri = ((2, 0), (0, 2), (4, 2))
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
    tri = ((2, 0), (0, 2), (4, 2))
    out = np.array([[0, 0, 3, 0, 0],
                    [0, 3, 3, 3, 0],
                    [3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle_interp(tri, (0, 4, 0, 5), (0, 0, 0), (agg,), (3, 3, 3))
    np.testing.assert_equal(agg, out)

    tri = ((2, 0), (0, 2), (4, 2))
    out = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 2, 0],
                    [2, 2, 2, 2, 3],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle_interp(tri, (0, 4, 0, 5), (0, 0, 0), (agg,), (1, 2, 3))
    np.testing.assert_equal(agg, out)

    tri = ((2, 0), (0, 2), (4, 2))
    out = np.array([[0, 0, 3, 0, 0],
                    [0, 4, 5, 6, 0],
                    [6, 6, 7, 8, 9],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle_interp(tri, (0, 4, 0, 5), (0, 0, 0), (agg,), (3, 6, 9))
    np.testing.assert_equal(agg, out)

    tri = ((2, 0), (0, 2), (4, 2))
    out = np.array([[0, 0, 6, 0, 0],
                    [0, 5, 4, 4, 0],
                    [4, 3, 3, 2, 2],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle_interp(tri, (0, 4, 0, 5), (0, 0, 0), (agg,), (6, 4, 2))
    np.testing.assert_equal(agg, out)

def test_draw_triangle_subpixel():
    """Assert that we draw subpixel triangles properly, both with and without
    interpolation.
    """
    # With interpolation
    tri = ((2, 0), (0, 2), (4, 2),
           (2, 3), (2, 3), (2, 3),
           (2, 3), (2, 3), (2, 3))
    out = np.array([[0, 0, 6, 0, 0],
                    [0, 5, 4, 4, 0],
                    [4, 3, 3, 2, 2],
                    [0, 0, 8, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle_interp(tri[:3], (0, 4, 0, 5), (0, 0, 0), (agg,), (6, 4, 2))
    draw_triangle_interp(tri[3:6], (2, 2, 3, 3), (0, 0, 0), (agg,), (6, 4, 2))
    draw_triangle_interp(tri[6:], (2, 2, 3, 3), (0, 0, 0), (agg,), (6, 4, 2))
    np.testing.assert_equal(agg, out)

    # Without interpolation
    tri = ((2, 0), (0, 2), (4, 2),
           (2, 3), (2, 3), (2, 3),
           (2, 3), (2, 3), (2, 3))
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

    _,pymax = map_onto_pixel_for_line((1.0, 0.0, sy, 0.0),
                                      (0.0, 1.0, 0.0, ymax),
                                      1.0, y)

    assert pymax==num_y_pixels-1


def test_lines_xy_validate():
    g = LinesAxis1(['x0', 'x1'], ['y11', 'y12'])
    g.validate(
        dshape("{x0: int32, x1: int32, y11: float32, y12: float32}"))

    with pytest.raises(ValueError):
        g.validate(
            dshape("{x0: int32, x1: float32, y11: string, y12: float32}"))
