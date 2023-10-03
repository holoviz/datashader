import pytest

from datashader.datashape import (
    promote, Option, float64, int64, float32, optionify,
    string, datetime_ as datetime, dshape)


def test_simple():
    x = int64
    y = float32
    z = promote(x, y)
    assert z == float64


def test_option():
    x = int64
    y = Option(float32)
    z = promote(x, y)
    assert z == Option(float64)


def test_no_promote_option():
    x = int64
    y = Option(float64)
    z = promote(x, y, promote_option=False)
    assert z == float64


def test_option_in_parent():
    x = int64
    y = Option(float32)
    z = optionify(x, y, y)
    assert z == y


@pytest.mark.parametrize('x,y,p,r',
                         [[string, string, True, string],
                          [string, string, False, string],

                          [Option(string),
                           Option(string),
                           True,
                           Option(string)],

                          [Option(string),
                           Option(string),
                           False,
                           Option(string)],

                          [Option(string),
                           string,
                           True,
                           Option(string)],

                          [Option(string),
                           string,
                           False,
                           string],

                          [Option(string),
                           dshape('?string'),
                           True,
                           Option(string)],

                          [dshape('?string'),
                           Option(string),
                           False,
                           Option(string)],

                          [dshape('string'),
                           Option(string),
                           True,
                           Option(string)],

                          [dshape('string'),
                           Option(string),
                           False,
                           string]])
def test_promote_string_with_option(x, y, p, r):
    assert (promote(x, y, promote_option=p) ==
            promote(y, x, promote_option=p) ==
            r)


@pytest.mark.parametrize('x,y,p,r',
                         [[datetime, datetime, True, datetime],
                          [datetime, datetime, False, datetime],

                          [Option(datetime),
                           Option(datetime),
                           True,
                           Option(datetime)],

                          [Option(datetime),
                           Option(datetime),
                           False,
                           Option(datetime)],

                          [Option(datetime),
                           datetime,
                           True,
                           Option(datetime)],

                          [Option(datetime),
                           datetime,
                           False,
                           datetime],

                          [Option(datetime),
                           dshape('?datetime'),
                           True,
                           Option(datetime)],

                          [dshape('?datetime'),
                           Option(datetime),
                           False,
                           Option(datetime)],

                          [dshape('datetime'),
                           Option(datetime),
                           True,
                           Option(datetime)],

                          [dshape('datetime'),
                           Option(datetime),
                           False,
                           datetime]])
def test_promote_datetime_with_option(x, y, p, r):
    assert (promote(x, y, promote_option=p) ==
            promote(y, x, promote_option=p) ==
            r)
