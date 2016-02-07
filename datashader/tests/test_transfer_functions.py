from io import BytesIO

import numpy as np
import xarray as xr
import PIL
import pytest

import datashader.transfer_functions as tf


coords = [np.array([0, 1, 2]), np.array([3, 4, 5])]
dims = ['y_axis', 'x_axis']

a = np.arange(10, 19, dtype='i4').reshape((3, 3))
a[[0, 1, 2], [0, 1, 2]] = 0
s_a = xr.DataArray(a, coords=coords, dims=dims)
b = np.arange(10, 19, dtype='f4').reshape((3, 3))
b[[0, 1, 2], [0, 1, 2]] = np.nan
s_b = xr.DataArray(b, coords=coords, dims=dims)
c = np.arange(10, 19, dtype='f8').reshape((3, 3))
c[[0, 1, 2], [0, 1, 2]] = np.nan
s_c = xr.DataArray(c, coords=coords, dims=dims)
agg = xr.Dataset(dict(a=s_a, b=s_b, c=s_c))


@pytest.mark.parametrize(['attr'], ['a', 'b', 'c'])
def test_interpolate(attr):
    x = getattr(agg, attr)
    img = tf.interpolate(x, 'pink', 'red', how='log')
    sol = np.array([[0, 4291543295, 4286741503],
                    [4283978751, 0, 4280492543],
                    [4279242751, 4278190335, 0]], dtype='u4')
    sol = xr.DataArray(sol, coords=coords, dims=dims)
    assert img.equals(sol)
    img = tf.interpolate(x, 'pink', 'red', how='cbrt')
    sol = np.array([[0, 4291543295, 4284176127],
                    [4282268415, 0, 4279834879],
                    [4278914047, 4278190335, 0]], dtype='u4')
    sol = xr.DataArray(sol, coords=coords, dims=dims)
    assert img.equals(sol)
    img = tf.interpolate(x, 'pink', 'red', how='linear')
    sol = np.array([[0, 4291543295, 4289306879],
                    [4287070463, 0, 4282597631],
                    [4280361215, 4278190335, 0]])
    sol = xr.DataArray(sol, coords=coords, dims=dims)
    assert img.equals(sol)
    img = tf.interpolate(x, 'pink', 'red', how=lambda x: x ** 2)
    sol = np.array([[0, 4291543295, 4291148543],
                    [4290030335, 0, 4285557503],
                    [4282268415, 4278190335, 0]], dtype='u4')
    sol = xr.DataArray(sol, coords=coords, dims=dims)
    assert img.equals(sol)


def test_colorize():
    coords = [np.array([0, 1]), np.array([2, 5])]
    cat_agg = xr.DataArray(np.array([[(0, 12, 0), (3, 0, 3)],
                                    [(12, 12, 12), (24, 0, 0)]]),
                           coords=(coords + [['a', 'b', 'c']]),
                           dims=(dims + ['cats']))

    colors = [(255, 0, 0), '#0000FF', 'orange']

    img = tf.colorize(cat_agg, colors, how='log')
    sol = np.array([[3137273856, 2449494783],
                    [4266997674, 3841982719]])
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert img.equals(sol)
    colors = dict(zip('abc', colors))
    img = tf.colorize(cat_agg, colors, how='cbrt')
    sol = np.array([[3070164992, 2499826431],
                    [4283774890, 3774873855]])
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert img.equals(sol)
    img = tf.colorize(cat_agg, colors, how='linear')
    sol = np.array([[1660878848, 989876991],
                    [4283774890, 2952790271]])
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert img.equals(sol)
    img = tf.colorize(cat_agg, colors, how=lambda x: x ** 2)
    sol = np.array([[788463616, 436228863],
                    [4283774890, 2080375039]])
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert img.equals(sol)


coords2 = [np.array([0, 2]), np.array([3, 5])]
img1 = tf.Image(np.dstack([np.array([[255, 0], [0, 125]], 'uint8'),
                           np.array([[255, 0], [0, 255]], 'uint8'),
                           np.array([[0, 0], [0, 0]], 'uint8'),
                           np.array([[255, 0], [0, 255]], 'uint8')])
                .view(np.uint32).reshape((2, 2)), coords=coords2, dims=dims)

img2 = tf.Image(np.dstack([np.array([[0, 0], [0, 255]], 'uint8'),
                           np.array([[0, 0], [0, 125]], 'uint8'),
                           np.array([[0, 0], [0, 125]], 'uint8'),
                           np.array([[0, 0], [255, 125]], 'uint8')])
                .view(np.uint32).reshape((2, 2)), coords=coords2, dims=dims)


def test_stack():
    img = tf.stack(img1, img2)
    assert (img.x_axis == img1.x_axis).all()
    assert (img.y_axis == img1.y_axis).all()
    chan = img.data.view([('r', 'uint8'), ('g', 'uint8'),
                         ('b', 'uint8'), ('a', 'uint8')])
    assert (chan['r'] == np.array([[255, 0], [0, 255]])).all()
    assert (chan['g'] == np.array([[255, 0], [0, 125]])).all()
    assert (chan['b'] == np.array([[0, 0], [0, 125]])).all()
    assert (chan['a'] == np.array([[255, 0], [255, 125]])).all()


def test_merge():
    img = tf.merge(img1, img2)
    assert (img.x_axis == img1.x_axis).all()
    assert (img.y_axis == img1.y_axis).all()
    chan = img.data.view([('r', 'uint8'), ('g', 'uint8'),
                         ('b', 'uint8'), ('a', 'uint8')])
    assert (chan['r'] == np.array([[127, 0], [0, 190]])).all()
    assert (chan['g'] == np.array([[127, 0], [0, 190]])).all()
    assert (chan['b'] == np.array([[0, 0], [0, 62]])).all()
    assert (chan['a'] == np.array([[127, 0], [127, 190]])).all()
    assert (tf.merge(img2, img1).data == img.data).all()


def test_Image_to_pil():
    img = img1.to_pil()
    assert isinstance(img, PIL.Image.Image)


def test_Image_to_bytesio():
    bytes = img1.to_bytesio()
    assert isinstance(bytes, BytesIO)
    assert bytes.tell() == 0
