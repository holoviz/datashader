import pytest

from datashape import dshape
from datashape.predicates import isfixed, _dimensions, isnumeric, isscalar
from datashape.coretypes import TypeVar, int32, Categorical


def test_isfixed():
    assert not isfixed(TypeVar('M') * int32)


def test_isscalar():
    assert isscalar('?int32')
    assert isscalar('float32')
    assert isscalar(int32)
    assert isscalar(Categorical(['a', 'b', 'c']))
    assert not isscalar('{a: int32, b: float64}')


def test_option():
    assert _dimensions('?int') == _dimensions('int')
    assert _dimensions('3 * ?int') == _dimensions('3 * int')


def test_time():
    assert not isnumeric('time')
