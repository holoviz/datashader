from datashape import dshape
import pandas as pd
import pytest

from datashader.glyphs import Point


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
