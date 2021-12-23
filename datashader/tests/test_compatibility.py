from __future__ import absolute_import
from pytest import raises

from datashader.compatibility import apply


def test_apply():
    f = lambda a, b, c=1, d=2: a + b + c + d
    assert apply(f, (1, 2,)) == 6
    assert apply(f, (1, 2,), dict(c=3)) == 8
