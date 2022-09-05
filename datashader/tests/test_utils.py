from __future__ import annotations
from datashape import dshape
import numpy as np
from xarray import DataArray

from datashader.utils import Dispatcher, apply, calc_res, isreal, orient_array


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


def test_apply():
    f = lambda a, b, c=1, d=2: a + b + c + d
    assert apply(f, (1, 2,)) == 6
    assert apply(f, (1, 2,), dict(c=3)) == 8


def test_calc_res():
    x = [5, 7]
    y = [0, 1]
    z = [[0, 0], [0, 0]]
    dims = ('y', 'x')

    # x and y increasing
    xarr = DataArray(z, coords=dict(x=x, y=y), dims=dims)
    xres, yres = calc_res(xarr)
    assert xres == 2
    assert yres == -1

    # x increasing, y decreasing
    xarr = DataArray(z, coords=dict(x=x, y=y[::-1]), dims=dims)
    xres, yres = calc_res(xarr)
    assert xres == 2
    assert yres == 1

    # x decreasing, y increasing
    xarr = DataArray(z, coords=dict(x=x[::-1], y=y), dims=dims)
    xres, yres = calc_res(xarr)
    assert xres == -2
    assert yres == -1

    # x and y decreasing
    xarr = DataArray(z, coords=dict(x=x[::-1], y=y[::-1]), dims=dims)
    xres, yres = calc_res(xarr)
    assert xres == -2
    assert yres == 1


def test_orient_array():
    x = [5, 7]
    y = [0, 1]
    z = np.array([[0, 1], [2, 3]])
    dims = ('y', 'x')

    # x and y increasing
    xarr = DataArray(z, coords=dict(x=x, y=y), dims=dims)
    arr = orient_array(xarr)
    assert np.array_equal(arr, z)

    # x increasing, y decreasing
    xarr = DataArray(z, coords=dict(x=x, y=y[::-1]), dims=dims)
    arr = orient_array(xarr)
    assert np.array_equal(arr, z[::-1])

    # x decreasing, y increasing
    xarr = DataArray(z, coords=dict(x=x[::-1], y=y), dims=dims)
    arr = orient_array(xarr)
    assert np.array_equal(arr, z[:, ::-1])

    # x and y decreasing
    xarr = DataArray(z, coords=dict(x=x[::-1], y=y[::-1]), dims=dims)
    arr = orient_array(xarr)
    assert np.array_equal(arr, z[::-1, ::-1])
