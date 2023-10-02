"""Testing the test helpers.

Kill me now.
"""
import pytest

from datashader.datashape.coretypes import (
    DateTime,
    R,
    String,
    Time,
    TimeDelta,
    Tuple,
    Option,
    int32,
    float32,
)
from datashader.datashape.util import dshape
from datashader.datashape.util.testing import assert_dshape_equal


def test_datashape_measure():
    assert_dshape_equal(dshape('int'), dshape('int'))

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(dshape('int'), dshape('string'))
    assert 'int32 != string' in str(e.value)
    assert '_.measure' in str(e.value)


def test_dim():
    assert_dshape_equal(dshape('var * int'), dshape('var * int'))
    assert_dshape_equal(dshape('3 * string'), dshape('3 * string'))

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(dshape('var * int'), dshape('3 * int'))
    assert 'var != 3' in str(e.value)
    assert '_.shape[0]' in str(e.value)

    assert_dshape_equal(dshape('var * var * int'), dshape('var * var * int'))
    assert_dshape_equal(dshape('var * 3 * string'), dshape('var * 3 * string'))
    assert_dshape_equal(
        dshape('3 * var * float32'),
        dshape('3 * var * float32'),
    )
    assert_dshape_equal(
        dshape('3 * 3 * datetime'),
        dshape('3 * 3 * datetime'),
    )

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(
            dshape('var * var * int'),
            dshape('3 * var * int'),
        )
    assert 'var != 3' in str(e.value)
    assert '_.shape[0]' in str(e.value)

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(
            dshape('var * var * int'),
            dshape('var * 3 * int'),
        )
    assert 'var != 3' in str(e.value)
    assert '_.shape[1]' in str(e.value)


def test_record():
    assert_dshape_equal(
        R['a': int32, 'b': float32],
        R['a': int32, 'b': float32],
    )

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(
            R['a': int32, 'b': float32],
            R['a': int32, 'b': int32],
        )
    assert "'float32' != 'int32'" in str(e)
    assert "_['b'].name" in str(e.value)

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(
            R['a': int32, 'b': float32],
            R['a': int32, 'c': float32],
        )
    assert "'b' != 'c'" in str(e.value)

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(
            R['b': float32, 'a': float32],
            R['a': int32, 'b': float32],
            check_record_order=False,
        )
    assert "'float32' != 'int32'" in str(e.value)
    assert "_['a']" in str(e.value)

    assert_dshape_equal(
        R['b': float32, 'a': int32],
        R['a': int32, 'b': float32],
        check_record_order=False,
    )

    # check a nested record with and without ordering
    assert_dshape_equal(
        R['a': R['b': float32, 'a': int32]],
        R['a': R['a': int32, 'b': float32]],
        check_record_order=False,
    )

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(
            R['a': R['a': int32, 'b': float32]],
            R['a': R['b': float32, 'a': int32]],
        )

    assert "'a' != 'b'" in str(e.value)
    assert "_['a']" in str(e.value)


def test_tuple():
    assert_dshape_equal(Tuple((int32, float32)), Tuple((int32, float32)))

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(Tuple((int32, float32)), Tuple((int32, int32)))
    assert "'float32' != 'int32'" in str(e)
    assert "_.dshapes[1].measure.name" in str(e.value)

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(Tuple((int32, float32)), Tuple((int32, int32)))
    assert "'float32' != 'int32'" in str(e)
    assert '_.dshapes[1].measure.name' in str(e.value)


def test_option():
    assert_dshape_equal(Option(int32), Option(int32))

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(Option(int32), Option(float32))
    assert "'int32' != 'float32'" in str(e.value)
    assert '_.ty' in str(e.value)


def test_string():
    assert_dshape_equal(String(), String())
    assert_dshape_equal(String('U8'), String('U8'))
    assert_dshape_equal(String(1), String(1))
    assert_dshape_equal(String(1, 'U8'), String(1, 'U8'))

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(String('U8'), String('U16'))

    assert "'U8' != 'U16'" in str(e.value)
    assert '_.encoding' in str(e.value)

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(String(1), String(2))
    assert '1 != 2' in str(e.value)
    assert '_.fixlen' in str(e.value)


def test_timedelta():
    assert_dshape_equal(TimeDelta(), TimeDelta())
    assert_dshape_equal(TimeDelta('ns'), TimeDelta('ns'))

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(TimeDelta('us'), TimeDelta('ns'))
    assert "'us' != 'ns'" in str(e.value)
    assert '_.unit' in str(e.value)

    assert_dshape_equal(
        TimeDelta('us'),
        TimeDelta('ns'),
        check_timedelta_unit=False,
    )


@pytest.mark.parametrize('cls', (DateTime, Time))
def test_datetime(cls):
    assert_dshape_equal(cls(), cls())
    assert_dshape_equal(cls('US/Eastern'), cls('US/Eastern'))

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(cls('US/Eastern'), cls('US/Central'))
    assert "'US/Eastern' != 'US/Central'" in str(e.value)
    assert '_.tz' in str(e.value)

    assert_dshape_equal(
        cls('US/Eastern'),
        cls('US/Central'),
        check_tz=False,
    )


def test_nested():
    assert_dshape_equal(
        dshape('var * {a: 3 * {b: int32}}'),
        dshape('var * {a: 3 * {b: int32}}'),
    )

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(
            dshape('var * {a: 3 * {b: int32}}'),
            dshape('var * {a: 3 * {b: float32}}'),
        )
    assert "'int32' != 'float32'" in str(e.value)
    assert "_.measure['a'].measure['b'].name" in str(e.value)


@pytest.mark.parametrize(
    'dshape_,contains', (
        (
            '(string, int64) -> int64', (
                'string != int32',
                '_.measure.argtypes[0].measure',
            ),
        ),
        (
            '(int32, int32) -> int64', (
                "'int32' != 'int64'",
                '_.measure.argtypes[1].measure.name',
            ),
        ),
        (
            '(int32, int64) -> int32', (
                "'int32' != 'int64'",
                '_.measure.restype.measure.name',
            ),
        ),
    ),
)
def test_function(dshape_, contains):
    base = dshape('(int32, int64) -> int64')
    assert_dshape_equal(base, base)

    with pytest.raises(AssertionError) as e:
        assert_dshape_equal(dshape(dshape_), base)
    for c in contains:
        assert c in str(e.value)
