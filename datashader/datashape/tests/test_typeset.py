from datashader import datashape
import pytest


def test_equal():
    assert datashape.integral == datashape.integral
    assert datashape.floating != datashape.integral


def test_repr():
    assert repr(datashape.integral) == '{integral}'


def test_custom_typeset_repr():
    mytypeset = datashape.TypeSet(datashape.int64, datashape.float64)
    assert repr(mytypeset).startswith('TypeSet(')
    assert repr(mytypeset).endswith('name=None)')


def test_register_already_existing_typeset_fails():
    mytypeset = datashape.TypeSet(datashape.int64, datashape.float64,
                                  name='foo')
    with pytest.raises(TypeError):
        datashape.typesets.register_typeset('foo', mytypeset)


def test_getitem():
    assert datashape.typesets.registry['integral'] == datashape.integral


def test_getitem_non_existent_typeset():
    with pytest.raises(KeyError):
        datashape.typesets.registry['footypeset']
