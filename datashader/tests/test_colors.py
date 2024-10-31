from __future__ import annotations
from datashader.colors import rgb, hex_to_rgb

import pytest


def test_hex_to_rgb():
    assert hex_to_rgb('#FAFBFC') == (250, 251, 252)
    with pytest.raises(ValueError):
        hex_to_rgb('#FFF')
    with pytest.raises(ValueError):
        hex_to_rgb('FFFFFF')
    with pytest.raises(ValueError):
        hex_to_rgb('#FFFFFG')


def test_rgb():
    assert rgb('#FAFBFC') == (250, 251, 252)
    assert rgb('blue') == (0, 0, 255)
    assert rgb((255, 255, 255)) == (255, 255, 255)
    with pytest.raises(ValueError):
        rgb((255, 256, 255))
    with pytest.raises(ValueError):
        rgb((-1, 255, 255))
    with pytest.raises(ValueError):
        rgb('foobar')
