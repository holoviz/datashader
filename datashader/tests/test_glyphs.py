from datashape import dshape
import pandas as pd
import numpy as np
import pytest

from datashader.glyphs import Point, _build_draw_line, _build_extend_line
from datashader.utils import ngjit


def test_point_bounds_check():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [5, 6, 7]})
    p = Point('x', 'y')
    assert p._compute_x_bounds(df) == (1, 3)
    assert p._compute_y_bounds(df) == (5, 7)


def test_point_validate():
    p = Point('x', 'y')
    p.validate(dshape("{x: int32, y: float32}"))
    with pytest.raises(ValueError):
        p.validate(dshape("{x: string, y: float32}"))


@ngjit
def append(i, x, y, agg):
    agg[y, x] += 1


def new_agg():
    return np.zeros((5, 5), dtype='i4')


mapper = ngjit(lambda x: x)
draw_line = _build_draw_line(append, mapper, mapper)
extend_line = _build_extend_line(draw_line)

bounds = (-3, 1, -3, 1)
vt = (1., 3., 1., 3.)


def test_draw_line():
    x0, y0 = (-3, -3)
    x1, y1 = (0, 0)
    out = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0]])
    agg = new_agg()
    draw_line(vt, bounds, x0, y0, x1, y1, 0, True, False, agg)
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_line(vt, bounds, x1, y1, x0, y0, 0, True, False, agg)
    np.testing.assert_equal(agg, out)
    # plot_start = False
    agg = new_agg()
    draw_line(vt, bounds, x0, y0, x1, y1, 0, False, False, agg)
    out[0, 0] = 0
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_line(vt, bounds, x1, y1, x0, y0, 0, False, False, agg)
    out[0, 0] = 1
    out[3, 3] = 0
    np.testing.assert_equal(agg, out)
    # Flip coords
    x0, y0 = (-3, 1)
    x1, y1 = (0, -2)
    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0]])
    agg = new_agg()
    draw_line(vt, bounds, x0, y0, x1, y1, 0, True, False, agg)
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_line(vt, bounds, x1, y1, x0, y0, 0, True, False, agg)
    np.testing.assert_equal(agg, out)
    # plot_start = False
    agg = new_agg()
    draw_line(vt, bounds, x0, y0, x1, y1, 0, False, False, agg)
    out[4, 0] = 0
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_line(vt, bounds, x1, y1, x0, y0, 0, False, False, agg)
    out[4, 0] = 1
    out[1, 3] = 0


def test_draw_line_same_point():
    x0, y0 = (0, 0)
    x1, y1 = (0.1, 0.1)
    agg = new_agg()
    draw_line(vt, bounds, x0, y0, x1, y1, 0, True, False, agg)
    assert agg.sum() == 2
    assert agg[3, 3] == 2
    agg = new_agg()
    draw_line(vt, bounds, x0, y0, x1, y1, 0, False, False, agg)
    assert agg.sum() == 1
    assert agg[3, 3] == 1
    agg = new_agg()
    draw_line(vt, bounds, x0, y0, x1, y1, 0, True, True, agg)
    assert agg.sum() == 1
    assert agg[3, 3] == 1


def test_draw_line_vertical_horizontal():
    # Vertical
    x0, y0 = (0, 0)
    x1, y1 = (0, -3)
    agg = new_agg()
    draw_line(vt, bounds, x0, y0, x1, y1, 0, True, False, agg)
    out = new_agg()
    out[:4, 3] = 1
    np.testing.assert_equal(agg, out)
    # Horizontal
    agg = new_agg()
    draw_line(vt, bounds, y0, x0, y1, x1, 0, True, False, agg)
    out = new_agg()
    out[3, :4] = 1
    np.testing.assert_equal(agg, out)


def test_extend_lines():
    xs = np.array([0, -2, -2, 0, 0])
    ys = np.array([-1,  -1,  1.1, 1.1, -1])
    out = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0]])
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
                    [1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0]])
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
    out = np.diag([1, 1, 0, 1, 1])
    np.testing.assert_equal(agg, out)
