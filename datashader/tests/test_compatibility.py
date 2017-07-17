from pytest import raises

from datashader.compatibility import apply, _exec


def test__exec():
    c = "def foo(a):\n    return bar(a) + 1"
    namespace = {'bar': lambda a: a + 1}
    bar = lambda a: a - 1  # noqa (define a different local ``bar`` to ensure
                           # that names are pulled from namespace, not locals)
    _exec(c, namespace)
    foo = namespace['foo']
    assert foo(1) == 3
    namespace = {}
    _exec(c, namespace)
    foo = namespace['foo']
    with raises(NameError):
        foo(1)


def test_apply():
    f = lambda a, b, c=1, d=2: a + b + c + d
    assert apply(f, (1, 2,)) == 6
    assert apply(f, (1, 2,), dict(c=3)) == 8
