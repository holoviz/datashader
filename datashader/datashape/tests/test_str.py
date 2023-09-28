import unittest
import pytest

import datashape
from datashape import dshape, DataShapeSyntaxError


class TestDataShapeStr(unittest.TestCase):
    def test_primitive_measure_str(self):
        self.assertEqual(str(datashape.int8), 'int8')
        self.assertEqual(str(datashape.int16), 'int16')
        self.assertEqual(str(datashape.int32), 'int32')
        self.assertEqual(str(datashape.int64), 'int64')
        self.assertEqual(str(datashape.uint8), 'uint8')
        self.assertEqual(str(datashape.uint16), 'uint16')
        self.assertEqual(str(datashape.uint32), 'uint32')
        self.assertEqual(str(datashape.uint64), 'uint64')
        self.assertEqual(str(datashape.float32), 'float32')
        self.assertEqual(str(datashape.float64), 'float64')
        self.assertEqual(str(datashape.string), 'string')
        self.assertEqual(str(datashape.String(3)), 'string[3]')
        self.assertEqual(str(datashape.String('A')), "string['A']")

    def test_structure_str(self):
        self.assertEqual(str(dshape('{x:int32, y:int64}')),
                         '{x: int32, y: int64}')

    def test_array_str(self):
        self.assertEqual(str(dshape('3*5*int16')),
                         '3 * 5 * int16')

    def test_primitive_measure_repr(self):
        self.assertEqual(repr(datashape.int8),      'ctype("int8")')
        self.assertEqual(repr(datashape.int16),     'ctype("int16")')
        self.assertEqual(repr(datashape.int32),     'ctype("int32")')
        self.assertEqual(repr(datashape.int64),     'ctype("int64")')
        self.assertEqual(repr(datashape.uint8),     'ctype("uint8")')
        self.assertEqual(repr(datashape.uint16),    'ctype("uint16")')
        self.assertEqual(repr(datashape.uint32),    'ctype("uint32")')
        self.assertEqual(repr(datashape.uint64),    'ctype("uint64")')
        self.assertEqual(repr(datashape.float32),   'ctype("float32")')
        self.assertEqual(repr(datashape.float64),   'ctype("float64")')
        self.assertEqual(repr(datashape.string),    'ctype("string")')
        self.assertEqual(repr(datashape.String(3)), 'ctype("string[3]")')
        self.assertEqual(repr(datashape.String('A')),
                         """ctype("string['A']")""")

    def test_structure_repr(self):
        self.assertEqual(repr(dshape('{x:int32, y:int64}')),
                         'dshape("{x: int32, y: int64}")')

    def test_array_repr(self):
        self.assertEqual(repr(dshape('3*5*int16')),
                         'dshape("3 * 5 * int16")')


@pytest.mark.parametrize('s',
                         ['{"./abc": int64}',
                          '{"./a b c": float64}',
                          '{"./a b\tc": string}',
                          '{"./a/[0 1 2]/b/\\n": float32}',
                          pytest.mark.xfail('{"/a/b/0/c\v/d": int8}',
                                            raises=DataShapeSyntaxError),
                          pytest.mark.xfail('{"/a/b/0/c\n/d": int8}',
                                            raises=DataShapeSyntaxError),
                          pytest.mark.xfail('{"/a/b/0/c\r/d": int8}',
                                            raises=DataShapeSyntaxError)])
def test_arbitrary_string(s):
    ds = dshape(s)
    assert dshape(str(ds)) == ds
