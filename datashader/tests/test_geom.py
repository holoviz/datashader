from math import inf
import numpy as np
from datashader.geom import (
    Points, PointsArray, Lines, LinesArray, Polygons, PolygonsArray
)

unit_square_cw = np.array([1, 1,  1, 2,  2, 2,  2, 1,  1, 1], dtype='float64')
large_square_ccw = np.array([0, 0, 3, 0, 3, 3, 0, 3, 0, 0], dtype='float64')
hole_sep = np.array([-inf, -inf])
fill_sep = np.array([inf, inf])


def test_points():
    points = Points(unit_square_cw)
    assert points.length == 0.0
    assert points.area == 0.0


def test_points_array():
    points = PointsArray([
        unit_square_cw,
        large_square_ccw,
        np.concatenate([large_square_ccw, hole_sep, unit_square_cw])
    ])

    np.testing.assert_equal(points.length, [0.0, 0.0, 0.0])
    np.testing.assert_equal(points.area, [0.0, 0.0, 0.0])


def test_lines():
    lines = Lines(unit_square_cw)
    assert lines.length == 4.0
    assert lines.area == 0.0


def test_lines_array():
    lines = LinesArray([
        unit_square_cw,
        large_square_ccw,
        np.concatenate([large_square_ccw, hole_sep, unit_square_cw])
    ])

    np.testing.assert_equal(lines.length, [4.0, 12.0, 16.0])
    np.testing.assert_equal(lines.area, [0.0, 0.0, 0.0])


def test_polygons():
    polygons = Polygons(np.concatenate([large_square_ccw, hole_sep, unit_square_cw]))
    assert polygons.length == 16.0
    assert polygons.area == 8.0


def test_polygons_array():
    polygons = PolygonsArray([
        large_square_ccw,
        np.concatenate([large_square_ccw, hole_sep, unit_square_cw]),
        unit_square_cw
    ])
    np.testing.assert_equal(polygons.length, [12.0, 16.0, 4.0])
    np.testing.assert_equal(polygons.area, [9.0, 8.0, -1.0])
