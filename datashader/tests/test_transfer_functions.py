from io import BytesIO

from dynd import nd
import numpy as np
import PIL
import pytest

import datashader.transfer_functions as tf


a = np.arange(10, 19, dtype='i4').reshape((3, 3))
a[[0, 1, 2], [0, 1, 2]] = 0
b = np.arange(10, 19, dtype='f8').reshape((3, 3))
b[[0, 1, 2], [0, 1, 2]] = np.nan
c = np.arange(10, 19, dtype='i8').reshape((3, 3))
c[[0, 1, 2], [0, 1, 2]] = np.iinfo('i8').min

agg = nd.empty((3, 3), '{a: int32, b: ?float64, c: ?int64}')
agg.a = a
agg.b = b
nd.as_numpy(agg.c.view_scalars(agg.c.dtype.value_type))[:] = c


@pytest.mark.parametrize(['attr'], ['a', 'b', 'c'])
def test_interpolate(attr):
    x = getattr(agg, attr)
    img = tf.interpolate(x, 'pink', 'red', how='log').img
    sol = np.array([[0, 4291543295, 4289043711],
                    [4286675711, 0, 4282268671],
                    [4280163839, 4278190335, 0]])
    assert (img == sol).all()
    img = tf.interpolate(x, 'pink', 'red', how='cbrt').img
    sol = np.array([[0, 4291543295, 4289109503],
                    [4286807295, 0, 4282399999],
                    [4280229375, 4278190335, 0]])
    assert (img == sol).all()
    img = tf.interpolate(x, 'pink', 'red', how='linear').img
    sol = np.array([[0, 4291543295, 4289306879],
                    [4287070463, 0, 4282597631],
                    [4280361215, 4278190335, 0]])
    assert (img == sol).all()
    img = tf.interpolate(x, 'pink', 'red', how=lambda x: x ** 2).img
    sol = np.array([[0, 4291543295, 4289504255],
                    [4287399423, 0, 4282992127],
                    [4280624127, 4278190335, 0]])
    assert (img == sol).all()


def test_colorize():
    colors = [(255, 0, 0), '#0000FF', 'orange']
    agg = np.array([[(0, 12, 0), (3, 0, 3)],
                    [(12, 12, 12), (24, 0, 0)]],
                   dtype=[(i, 'uint8') for i in 'abc'])
    agg = nd.asarray(agg)
    img = tf.colorize(agg, colors, how='log').img
    sol = np.array([[3137273856, 2449494783],
                    [4266997674, 3841982719]])
    assert (img == sol).all()
    colors = dict(zip('abc', colors))
    img = tf.colorize(agg, colors, how='cbrt').img
    sol = np.array([[3070164992, 2499826431],
                    [4283774890, 3774873855]])
    assert (img == sol).all()
    img = tf.colorize(agg, colors, how='linear').img
    sol = np.array([[1660878848, 989876991],
                    [4283774890, 2952790271]])
    assert (img == sol).all()
    img = tf.colorize(agg, colors, how=lambda x: x ** 2).img
    sol = np.array([[788463616, 436228863],
                    [4283774890, 2080375039]])
    assert (img == sol).all()


img1 = tf.Image(np.dstack([np.array([[255, 0], [0, 125]], 'uint8'),
                           np.array([[255, 0], [0, 255]], 'uint8'),
                           np.array([[0, 0], [0, 0]], 'uint8'),
                           np.array([[255, 0], [0, 255]], 'uint8')])
                .view(np.uint32).reshape((2, 2)))

img2 = tf.Image(np.dstack([np.array([[0, 0], [0, 255]], 'uint8'),
                           np.array([[0, 0], [0, 125]], 'uint8'),
                           np.array([[0, 0], [0, 125]], 'uint8'),
                           np.array([[0, 0], [255, 125]], 'uint8')])
                .view(np.uint32).reshape((2, 2)))


def test_stack():
    img = tf.stack(img1, img2)
    chan = img.img.view([('r', 'uint8'), ('g', 'uint8'),
                         ('b', 'uint8'), ('a', 'uint8')])
    assert (chan['r'] == np.array([[255, 0], [0, 255]])).all()
    assert (chan['g'] == np.array([[255, 0], [0, 125]])).all()
    assert (chan['b'] == np.array([[0, 0], [0, 125]])).all()
    assert (chan['a'] == np.array([[255, 0], [255, 125]])).all()


def test_merge():
    img = tf.merge(img1, img2)
    chan = img.img.view([('r', 'uint8'), ('g', 'uint8'),
                         ('b', 'uint8'), ('a', 'uint8')])
    assert (chan['r'] == np.array([[127, 0], [0, 190]])).all()
    assert (chan['g'] == np.array([[127, 0], [0, 190]])).all()
    assert (chan['b'] == np.array([[0, 0], [0, 62]])).all()
    assert (chan['a'] == np.array([[127, 0], [127, 190]])).all()
    assert (tf.merge(img2, img1).img == img.img).all()


def test_Image_to_pil():
    img = img1.to_pil()
    assert isinstance(img, PIL.Image.Image)


def test_Image_to_bytesio():
    bytes = img1.to_bytesio()
    assert isinstance(bytes, BytesIO)
    assert bytes.tell() == 0
