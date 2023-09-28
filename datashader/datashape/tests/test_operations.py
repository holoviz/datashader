import datashape
import pytest


def test_scalar_subarray():
    assert datashape.int32.subarray(0) == datashape.int32
    with pytest.raises(IndexError):
        datashape.int32.subarray(1)
    assert datashape.string.subarray(0) == datashape.string
    with pytest.raises(IndexError):
        datashape.string.subarray(1)


def test_array_subarray():
    assert (datashape.dshape('3 * int32').subarray(0) ==
            datashape.dshape('3 * int32'))
    assert (datashape.dshape('3 * int32').subarray(1) ==
            datashape.DataShape(datashape.int32))
    assert (str(datashape.dshape('3 * var * M * int32').subarray(2)) ==
            str(datashape.dshape('M * int32')))
    assert (str(datashape.dshape('3 * var * M * float64').subarray(3)) ==
            str(datashape.float64))


def test_dshape_compare():
    assert datashape.int32 != datashape.dshape('1 * int32')
