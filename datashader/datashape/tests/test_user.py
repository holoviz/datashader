import pytest

from datashader.datashape.user import issubschema, validate
from datashader.datashape import dshape
from datetime import date, time, datetime
import numpy as np


min_np = pytest.mark.skipif(
    np.__version__ > '1.14',
    reason="issubdtype no longer downcasts"
)


@min_np
def test_validate():
    assert validate(int, 1)
    assert validate('int', 1)
    assert validate(str, 'Alice')
    assert validate(dshape('string'), 'Alice')
    assert validate(dshape('int'), 1)
    assert validate(dshape('int')[0], 1)
    assert validate('real', 2.0)
    assert validate('2 * int', (1, 2))
    assert not validate('3 * int', (1, 2))
    assert not validate('2 * int', 2)


@min_np
def test_nested_iteratables():
    assert validate('2 * 3 * int', [(1, 2, 3), (4, 5, 6)])


def test_numeric_tower():
    assert validate(np.integer, np.int32(1))
    assert validate(np.number, np.int32(1))


@min_np
def test_validate_dicts():
    assert validate('{x: int, y: int}', {'x': 1, 'y': 2})
    assert not validate('{x: int, y: int}', {'x': 1, 'y': 2.0})
    assert not validate('{x: int, y: int}', {'x': 1, 'z': 2})

    assert validate('var * {x: int, y: int}', [{'x': 1, 'y': 2}])

    assert validate('var * {x: int, y: int}', [{'x': 1, 'y': 2},
                                               {'x': 3, 'y': 4}])


@min_np
def test_tuples_can_be_records_too():
    assert validate('{x: int, y: real}', (1, 2.0))
    assert not validate('{x: int, y: real}', (1.0, 2))


def test_datetimes():
    assert validate('time', time(12, 0, 0))
    assert validate('date', date(1999, 1, 20))
    assert validate('datetime', datetime(1999, 1, 20, 12, 0, 0))


def test_numpy():
    assert validate('2 * int32', np.array([1, 2], dtype='int32'))


def test_issubschema():
    assert issubschema('int', 'int')
    assert not issubschema('int', 'float32')

    assert issubschema('2 * int', '2 * int')
    assert not issubschema('2 * int', '3 * int')

    # assert issubschema('float32', 'real')


def test_integration():
    assert validate('{name: string, arrived: date}',
                    {'name': 'Alice', 'arrived': date(2012, 1, 5)})
