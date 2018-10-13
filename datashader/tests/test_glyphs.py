from datashape import dshape
import pandas as pd
import numpy as np
import pytest

from datashader.glyphs import (Point, _build_draw_line, _build_map_onto_pixel_for_line,
                               _build_extend_line, _build_draw_triangle,
                               _build_map_onto_pixel_for_triangle, _build_extend_triangles)
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
draw_line = _build_draw_line(append)
extend_line = _build_extend_line(draw_line, map_onto_pixel_for_line)

# Triangles rasterization
draw_triangle, draw_triangle_interp = _build_draw_triangle(tri_append)
extend_triangles = _build_extend_triangles(draw_triangle, draw_triangle_interp, map_onto_pixel_for_triangle)

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

def test_draw_triangle_nointerp():
    """Assert that we draw triangles properly, without interpolation enabled.
    """
    # Isosceles triangle
    tri = [(2, 0), (0, 2), (4, 2)]
    out = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (0, 4, 0, 5), (0, 0, 0), agg, 1)
    np.testing.assert_equal(agg, out)

    # Right triangle
    tri = [(2, 0), (0, 2), (2, 2)]
    out = np.array([[0, 0, 2, 0, 0],
                    [0, 2, 2, 0, 0],
                    [2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (0, 4, 0, 5), (0, 0, 0), agg, 2)
    np.testing.assert_equal(agg, out)

    # Two right trimesh
    tri = [(2, 0), (1, 1), (2, 1),
           (2, 1), (2, 2), (3, 2)]
    out = np.array([[0, 0, 3, 0, 0],
                    [0, 3, 6, 0, 0],
                    [0, 0, 3, 3, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri[:3], (0, 4, 0, 5), (0, 0, 0), agg, 3)
    draw_triangle(tri[3:], (0, 4, 0, 5), (0, 0, 0), agg, 3)
    np.testing.assert_equal(agg, out)

    # Draw isoc triangle with clipping
    tri = [(2, 0), (0, 2), (4, 2)]
    out = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (0, 3, 0, 2), (0, 0, 0), agg, 1)
    np.testing.assert_equal(agg, out)
    # clip from right and left
    out = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (1, 3, 0, 2), (0, 0, 0), agg, 1)
    np.testing.assert_equal(agg, out)
    # clip from right, left, top
    out = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (1, 3, 1, 2), (0, 0, 0), agg, 1)
    np.testing.assert_equal(agg, out)
    # clip from right, left, top, bottom
    out = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (1, 3, 1, 1), (0, 0, 0), agg, 1)
    np.testing.assert_equal(agg, out)

def test_draw_triangle_interp():
    """Assert that we draw triangles properly, with interpolation enabled.
    """
    # Isosceles triangle
    tri = [(2, 0), (0, 2), (4, 2)]
    out = np.array([[0, 0, 3, 0, 0],
                    [0, 3, 3, 3, 0],
                    [3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle_interp(tri, (0, 4, 0, 5), (0, 0, 0), agg, (3, 3, 3))
    np.testing.assert_equal(agg, out)

    tri = [(2, 0), (0, 2), (4, 2)]
    out = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 2, 0],
                    [2, 2, 2, 2, 3],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle_interp(tri, (0, 4, 0, 5), (0, 0, 0), agg, (1, 2, 3))
    np.testing.assert_equal(agg, out)

    tri = [(2, 0), (0, 2), (4, 2)]
    out = np.array([[0, 0, 3, 0, 0],
                    [0, 4, 5, 6, 0],
                    [6, 6, 7, 8, 9],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle_interp(tri, (0, 4, 0, 5), (0, 0, 0), agg, (3, 6, 9))
    np.testing.assert_equal(agg, out)

    tri = [(2, 0), (0, 2), (4, 2)]
    out = np.array([[0, 0, 6, 0, 0],
                    [0, 5, 4, 4, 0],
                    [4, 3, 3, 2, 2],
                    [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle_interp(tri, (0, 4, 0, 5), (0, 0, 0), agg, (6, 4, 2))
    np.testing.assert_equal(agg, out)

def test_draw_triangle_subpixel():
    """Assert that we draw subpixel triangles properly, both with and without
    interpolation.
    """
    # With interpolation
    tri = [(2, 0), (0, 2), (4, 2),
           (2, 3), (2, 3), (2, 3),
           (2, 3), (2, 3), (2, 3)]
    out = np.array([[0, 0, 6, 0, 0],
                    [0, 5, 4, 4, 0],
                    [4, 3, 3, 2, 2],
                    [0, 0, 8, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle_interp(tri[:3], (0, 4, 0, 5), (0, 0, 0), agg, (6, 4, 2))
    draw_triangle_interp(tri[3:6], (2, 2, 3, 3), (0, 0, 0), agg, (6, 4, 2))
    draw_triangle_interp(tri[6:], (2, 2, 3, 3), (0, 0, 0), agg, (6, 4, 2))
    np.testing.assert_equal(agg, out)

    # Without interpolation
    tri = [(2, 0), (0, 2), (4, 2),
           (2, 3), (2, 3), (2, 3),
           (2, 3), (2, 3), (2, 3)]
    out = np.array([[0, 0, 2, 0, 0],
                    [0, 2, 2, 2, 0],
                    [2, 2, 2, 2, 2],
                    [0, 0, 4, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri[:3], (0, 4, 0, 5), (0, 0, 0), agg, 2)
    draw_triangle(tri[3:6], (2, 2, 3, 3), (0, 0, 0), agg, 2)
    draw_triangle(tri[6:], (2, 2, 3, 3), (0, 0, 0), agg, 2)
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
