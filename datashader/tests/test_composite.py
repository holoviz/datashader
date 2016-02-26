import numpy as np

from datashader.composite import add, saturate, over, source

src = np.array([[0x00000000, 0x00ffffff, 0xffffffff],
                [0x7dff0000, 0x7d00ff00, 0x7d0000ff],
                [0xffff0000, 0xff000000, 0x3a3b3c3d]], dtype='uint32')

clear = np.uint32(0)
clear_black = np.uint32(0x00ffffff)
black = np.uint32(0xffffffff)
blue = np.uint32(0xffff0000)
half_blue = np.uint32(0x7dff0000)
half_purple = np.uint32(0x7d7d007d)


def test_source():
    o = src.copy()
    o[0, :2] = clear
    np.testing.assert_equal(source(src, clear), o)
    o[0, :2] = clear_black
    np.testing.assert_equal(source(src, clear_black), o)
    o[0, :2] = half_blue
    np.testing.assert_equal(source(src, half_blue), o)


def test_over():
    o = src.copy()
    o[0, 1] = 0
    np.testing.assert_equal(over(src, clear), o)
    np.testing.assert_equal(over(src, clear_black), o)
    o = np.array([[0xffffffff, 0xffffffff, 0xffffffff],
                  [0xffff8282, 0xff82ff82, 0xff8282ff],
                  [0xffff0000, 0xff000000, 0xffd2d2d2]])
    np.testing.assert_equal(over(src, black), o)
    o = np.array([[0xffff0000, 0xffff0000, 0xffffffff],
                  [0xffff0000, 0xff827d00, 0xff82007d],
                  [0xffff0000, 0xff000000, 0xffd20d0d]])
    np.testing.assert_equal(over(src, blue), o)
    o = np.array([[0x7dff0000, 0x7dff0000, 0xffffffff],
                  [0xbcff0000, 0xbc56a800, 0xbc5600a8],
                  [0xffff0000, 0xff000000, 0x9ab51616]])
    np.testing.assert_equal(over(src, half_blue), o)
    o = np.array([[0x7d7d007d, 0x7d7d007d, 0xffffffff],
                  [0xbcd3002a, 0xbc2aa82a, 0xbc2a00d3],
                  [0xffff0000, 0xff000000, 0x9a641664]])
    np.testing.assert_equal(over(src, half_purple), o)


def test_add():
    o = src.copy()
    o[0, 1] = 0
    np.testing.assert_equal(add(src, clear), o)
    np.testing.assert_equal(add(src, clear_black), o)
    o = np.array([[0xffffffff, 0xffffffff, 0xfffffffe],
                  [0xff7cffff, 0xffff7cff, 0xffffff7c],
                  [0xfffeffff, 0xffffffff, 0xff0d0d0c]])
    np.testing.assert_equal(add(src, black), o)
    o = np.array([[0xffff0000, 0xffff0000, 0xfffeffff],
                  [0xff7c0000, 0xffff7d00, 0xffff007d],
                  [0xfffe0000, 0xffff0000, 0xff0c0d0d]])
    np.testing.assert_equal(add(src, blue), o)
    o = np.array([[0x7dff0000, 0x7dff0000, 0xff7cffff],
                  [0xfaff0000, 0xfa7f7f00, 0xfa7f007f],
                  [0xff7c0000, 0xff7d0000, 0xb7c01313]])
    np.testing.assert_equal(add(src, half_blue), o)
    o = np.array([[0x7d7d007d, 0x7d7d007d, 0xff3cff3c],
                  [0xfabe003e, 0xfa3e7f3e, 0xfa3e00be],
                  [0xff3c003d, 0xff3d003d, 0xb7681368]])
    np.testing.assert_equal(add(src, half_purple), o)


def test_saturate():
    o = src.copy()
    o[0, 1] = 0
    np.testing.assert_equal(saturate(src, clear), o)
    np.testing.assert_equal(saturate(src, clear_black), o)
    o = np.full((3, 3), black, dtype='uint32')
    np.testing.assert_equal(saturate(src, black), o)
    o = np.full((3, 3), blue, dtype='uint32')
    np.testing.assert_equal(saturate(src, blue), o)
    o = np.array([[0x7dff0000, 0x7dff0000, 0xffff8282],
                  [0xfaff0000, 0xfa7f7f00, 0xfa7f007f],
                  [0xffff0000, 0xff7d0000, 0xb7c01313]])
    np.testing.assert_equal(saturate(src, half_blue), o)
    o = np.array([[0x7d7d007d, 0x7d7d007d, 0xffbf82bf],
                  [0xfabe003e, 0xfa3e7f3e, 0xfa3e00be],
                  [0xffbf003d, 0xff3d003d, 0xb7681368]])
    np.testing.assert_equal(saturate(src, half_purple), o)
