from itertools import starmap
import sys
from warnings import catch_warnings, simplefilter

import numpy as np
import pytest

from datashape.discovery import (discover, null, unite_identical, unite_base,
                                 unite_merge_dimensions, do_one,
                                 lowest_common_dshape)
from datashape.coretypes import (int64, float64, complex128, string, bool_,
                                 Tuple, Record, date_, datetime_, time_,
                                 timedelta_, int32, var, Option, real, Null,
                                 TimeDelta, String, float32, R)
from datashape.py2help import PY2, CPYTHON, mappingproxy, OrderedDict
from datashape.util.testing import assert_dshape_equal
from datashape import dshape
from datetime import date, time, datetime, timedelta


def test_simple():
    assert discover(3) == int64
    assert discover(3.0) == float64
    assert discover(3.0 + 1j) == complex128
    assert discover('Hello') == string
    assert discover(True) == bool_
    assert discover(None) == null


def test_long():
    if sys.version_info[0] == 2:
        assert eval('discover(3L)') == int64


def test_list():
    assert discover([1, 2, 3]) == 3 * discover(1)
    assert discover([1.0, 2.0, 3.0]) == 3 * discover(1.0)


def test_set():
    assert discover(set([1])) == 1 * discover(1)


def test_frozenset():
    assert discover(frozenset([1])) == 1 * discover(1)


def test_heterogeneous_ordered_container():
    assert discover(('Hello', 1)) == Tuple([discover('Hello'), discover(1)])


def test_string():
    assert discover('1') == discover(1)
    assert discover('1.0') == discover(1.0)
    assert discover('True') == discover(True)
    assert discover('true') == discover(True)


def test_record():
    assert (discover({'name': 'Alice', 'amount': 100}) ==
            Record([['amount', discover(100)],
                    ['name', discover('Alice')]]))


@pytest.mark.skipif(
    PY2 and not CPYTHON,
    reason='We cannot create mapping proxies in python 2 when not in CPython')
def test_mappingproxy():
    d = {'a': np.int64(1), 'b': 'cs', 'c': np.float32(1.0)}
    assert_dshape_equal(
        discover(mappingproxy(d)),
        discover(d),
    )


def test_ordereddict():
    od = OrderedDict((('c', np.int64(1)), ('b', 'cs'), ('a', np.float32(1.0))))
    assert_dshape_equal(
        discover(od),
        R['c': int64, 'b': string, 'a': float32],
    )


def test_datetime():
    inputs = ["1991-02-03 04:05:06",
              "11/12/1822 06:47:26.00",
              "1822-11-12T06:47:26",
              "Fri Dec 19 15:10:11 1997",
              "Friday, November 11, 2005 17:56:21",
              "1982-2-20 5:02:00",
              "20030331 05:59:59.9",
              "Jul  6 2030  5:55PM",
              "1994-10-20 T 11:15",
              "2013-03-04T14:38:05.123",
              datetime(2014, 1, 1, 12, 1, 1),
              # "15MAR1985:14:15:22",
              # "201303041438"
              ]
    for dt in inputs:
        assert discover(dt) == datetime_


def test_string_date():
    assert discover('2014-01-01') == date_


def test_python_date():
    assert discover(date(2014, 1, 1)) == date_


def test_single_space_string_is_not_date():
    assert discover(' ') == string


def test_string_that_looks_like_date():
    # GH 91
    assert discover("31-DEC-99 12.00.00.000000000") == string


def test_time():
    assert discover(time(12, 0, 1)) == time_


def test_timedelta():
    objs = starmap(timedelta, (range(10, 10 - i, -1) for i in range(1, 8)))
    for ts in objs:
        assert discover(ts) == timedelta_


def test_timedelta_strings():
    inputs = ["1 day",
              "-2 hours",
              "3 seconds",
              "1 microsecond",
              "1003 milliseconds"]
    for ts in inputs:
        assert discover(ts) == TimeDelta(unit=ts.split()[1])

    with pytest.raises(ValueError):
        TimeDelta(unit='buzz light-years')


def test_time_string():
    assert discover('12:00:01') == time_
    assert discover('12:00:01.000') == time_
    assert discover('12:00:01.123456') == time_
    assert discover('12:00:01.1234') == time_
    assert discover('10-10-01T12:00:01') == datetime_
    assert discover('10-10-01 12:00:01') == datetime_


def test_integrative():
    data = [{'name': 'Alice', 'amount': '100'},
            {'name': 'Bob', 'amount': '200'},
            {'name': 'Charlie', 'amount': '300'}]

    assert (dshape(discover(data)) ==
            dshape('3 * {amount: int64, name: string}'))


def test_numpy_scalars():
    assert discover(np.int32(1)) == int32
    assert discover(np.float64(1)) == float64


def test_numpy_array():
    assert discover(np.ones((3, 2), dtype=np.int32)) == dshape('3 * 2 * int32')


def test_numpy_array_with_strings():
    x = np.array(['Hello', 'world'], dtype='O')
    assert discover(x) == 2 * string


def test_numpy_recarray_with_strings():
    x = np.array([('Alice', 1), ('Bob', 2)],
                 dtype=[('name', 'O'), ('amt', 'i4')])
    assert discover(x) == dshape('2 * {name: string, amt: int32}')


unite = do_one([unite_identical,
                unite_merge_dimensions,
                unite_base])


def test_unite():
    assert unite([int32, int32, int32]) == 3 * int32
    assert unite([3 * int32, 2 * int32]) == 2 * (var * int32)
    assert unite([2 * int32, 2 * int32]) == 2 * (2 * int32)
    assert unite([3 * (2 * int32), 2 * (2 * int32)]) == 2 * (var * (2 * int32))


def test_unite_missing_values():
    assert unite([int32, null, int32]) == 3 * Option(int32)
    assert unite([string, null, int32])


def test_unite_tuples():
    assert (discover([[1, 1, 'hello'],
                     [1, '', ''],
                     [1, 1, 'hello']]) ==
            3 * Tuple([int64, Option(int64), Option(string)]))

    assert (discover([[1, 1, 'hello', 1],
                     [1, '', '', 1],
                     [1, 1, 'hello', 1]]) ==
            3 * Tuple([int64, Option(int64), Option(string), int64]))


def test_unite_records():
    assert (discover([{'name': 'Alice', 'balance': 100},
                     {'name': 'Bob', 'balance': ''}]) ==
            2 * Record([['balance', Option(int64)], ['name', string]]))

    assert (discover([{'name': 'Alice', 's': 'foo'},
                     {'name': 'Bob', 's': None}]) ==
            2 * Record([['name', string], ['s', Option(string)]]))

    assert (discover([{'name': 'Alice', 's': 'foo', 'f': 1.0},
                     {'name': 'Bob', 's': None, 'f': None}]) ==
            2 * Record([['f', Option(float64)],
                        ['name', string],
                        ['s', Option(string)]]))

    # assert unite((Record([['name', string], ['balance', int32]]),
    #               Record([['name', string]]))) == \
    #                 Record([['name', string], ['balance', Option(int32)]])


def test_dshape_missing_data():
    assert (discover([[1, 2, '', 3],
                     [1, 2, '', 3],
                     [1, 2, '', 3]]) ==
            3 * Tuple([int64, int64, null, int64]))


def test_discover_mixed():
    i = discover(1)
    f = discover(1.0)
    exp = 10 * Tuple([i, i, f, f])
    assert dshape(discover([[1, 2, 1.0, 2.0]] * 10)) == exp

    exp = 10 * (4 * f)
    assert dshape(discover([[1, 2, 1.0, 2.0], [1.0, 2.0, 1, 2]] * 5)) == exp


def test_test():
    expected = 2 * Tuple([string, int64])
    assert discover([['Alice', 100], ['Bob', 200]]) == expected


def test_discover_appropriate():
    assert discover((1, 1.0)) == Tuple([int64, real])
    assert discover([(1, 1.0), (1, 1.0), (1, 1)]) == 3 * Tuple([int64, real])


def test_big_discover():
    data = [['1'] + ['hello']*20] * 10
    assert discover(data) == 10 * Tuple([int64] + [string]*20)


def test_unite_base():
    assert unite_base([date_, datetime_]) == 2 * datetime_


def test_list_of_dicts_no_difference():
    data = [{'name': 'Alice', 'amount': 100},
            {'name': 'Bob'}]
    result = discover(data)
    expected = dshape('2 * {amount: ?int64, name: string}')
    assert result == expected


def test_list_of_dicts_difference():
    data = [{'name': 'Alice', 'amount': 100},
            {'name': 'Bob', 'house_color': 'blue'}]
    result = discover(data)
    s = '2 * {amount: ?int64, house_color: ?string, name: string}'
    expected = dshape(s)
    assert result == expected


def test_unite_base_on_records():
    dshapes = [dshape('{name: string, amount: int32}'),
               dshape('{name: string, amount: int32}')]
    assert unite_base(dshapes) == dshape('2 * {name: string, amount: int32}')

    dshapes = [Null(), dshape('{name: string, amount: int32}')]
    assert unite_base(dshapes) == dshape('2 * ?{name: string, amount: int32}')

    dshapes = [dshape('{name: string, amount: int32}'),
               dshape('{name: string, amount: int64}')]
    assert unite_base(dshapes) == dshape('2 * {name: string, amount: int64}')


def test_nested_complex_record_type():
    dt = np.dtype([('a', 'U7'), ('b', [('c', 'int64', 2), ('d', 'float64')])])
    x = np.zeros(5, dt)
    s = "5 * {a: string[7, 'U32'], b: {c: 2 * int64, d: float64}}"
    assert discover(x) == dshape(s)


def test_letters_only_strings():
    strings = ('sunday', 'monday', 'tuesday', 'wednesday', 'thursday',
               'friday', 'saturday', 'a', 'b', 'now', 'yesterday', 'tonight')
    for s in strings:
        assert discover(s) == string


def test_discover_array_like():
    class MyArray(object):
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    with catch_warnings(record=True) as wl:
        simplefilter('always')
        assert discover(MyArray((4, 3), 'f4')) == dshape('4 * 3 * float32')
    assert len(wl) == 1
    assert issubclass(wl[0].category, DeprecationWarning)
    assert 'MyArray' in str(wl[0].message)


@pytest.mark.xfail(sys.version_info[0] == 2,
                   raises=AssertionError,
                   reason=('discovery behavior is different for raw strings '
                           'in python 2'))
def test_discover_bytes():
    x = b'abcdefg'
    assert discover(x) == String('A')


def test_discover_undiscoverable():
    class MyClass(object):
        pass
    with pytest.raises(NotImplementedError):
        discover(MyClass())


@pytest.mark.parametrize('seq', [(), [], set()])
def test_discover_empty_sequence(seq):
    assert discover(seq) == var * string


@pytest.mark.xfail(raises=ValueError, reason='Not yet implemented')
def test_lowest_common_dshape_varlen_strings():
    assert lowest_common_dshape([String(10), String(11)]) == String(11)
    assert lowest_common_dshape([String(11), string]) == string


def test_discover_mock():
    try:
        from unittest.mock import Mock
    except ImportError:
        from mock import Mock

    # This used to segfault because we were sending mocks into numpy
    with pytest.raises(NotImplementedError):
        discover(Mock())


def test_string_with_overflow():
    assert discover('INF US Equity') == string
