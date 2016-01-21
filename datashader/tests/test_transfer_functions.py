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
    sol_log = np.array([[0, 4291543295, 4289043711],
                        [4286675711, 0, 4282268671],
                        [4280163839, 4278190335, 0]])
    assert (img == sol_log).all()
    img = tf.interpolate(x, 'pink', 'red', how='cbrt').img
    sol_cbrt = np.array([[0, 4291543295, 4289109503],
                         [4286807295, 0, 4282399999],
                         [4280229375, 4278190335, 0]])
    assert (img == sol_cbrt).all()
    img = tf.interpolate(x, 'pink', 'red', how='linear').img
    sol_lin = np.array([[0, 4291543295, 4289306879],
                        [4287070463, 0, 4282597631],
                        [4280361215, 4278190335, 0]])
    assert (img == sol_lin).all()


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
