from datashape import dshape

from datashader.utils import _exec, Dispatcher, isreal

from pytest import raises


def test__exec():
    c = "def foo(a):\n    return bar(a) + 1"
    namespace = {'bar': lambda a: a + 1}
    # Define a different local ``bar`` to ensure that names are pulled from
    # namespace, not locals
    bar = lambda a: a - 1
    _exec(c, namespace)
    foo = namespace['foo']
    assert foo(1) == 3
    namespace = {}
    _exec(c, namespace)
    foo = namespace['foo']
    with raises(NameError):
        foo(1)


def test_Dispatcher():
    foo = Dispatcher()
    foo.register(int, lambda a, b, c=1: a + b + c)
    foo.register(float, lambda a, b, c=1: a - b + c)
    foo.register(object, lambda a, b, c=1: 10)

    class Bar(object):
        pass
    b = Bar()
    assert foo(1, 2) == 4
    assert foo(1, 2.0, 3.0) == 6.0
    assert foo(1.0, 2.0, 3.0) == 2.0
    assert foo(b, 2) == 10


def test_isreal():
    assert isreal('int32')
    assert isreal(dshape('int32'))
    assert isreal('?int32')
    assert isreal('float64')
    assert not isreal('complex64')
    assert not isreal('{x: int64, y: float64}')
