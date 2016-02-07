from datashape import dshape

from datashader.utils import Dispatcher, isreal


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
