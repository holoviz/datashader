
import ctypes
import unittest

import pytest

from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record


class TestDataShapeCreation(unittest.TestCase):

    def test_raise_on_bad_input(self):
        # Make sure it raises exceptions on a few nonsense inputs
        self.assertRaises(TypeError, dshape, None)
        self.assertRaises(TypeError, dshape, lambda x: x+1)
        # Check issue 11
        self.assertRaises(datashape.DataShapeSyntaxError, dshape, '1 *')
        self.assertRaises(datashape.DataShapeSyntaxError, dshape, '1,')

    def test_reserved_future_bigint(self):
        # The "bigint" datashape is reserved for a future big integer type
        self.assertRaises(Exception, dshape, "bigint")

    def test_atom_shapes(self):
        self.assertEqual(dshape('bool'), dshape(datashape.bool_))
        self.assertEqual(dshape('int8'), dshape(datashape.int8))
        self.assertEqual(dshape('int16'), dshape(datashape.int16))
        self.assertEqual(dshape('int32'), dshape(datashape.int32))
        self.assertEqual(dshape('int64'), dshape(datashape.int64))
        self.assertEqual(dshape('uint8'), dshape(datashape.uint8))
        self.assertEqual(dshape('uint16'), dshape(datashape.uint16))
        self.assertEqual(dshape('uint32'), dshape(datashape.uint32))
        self.assertEqual(dshape('uint64'), dshape(datashape.uint64))
        self.assertEqual(dshape('float32'), dshape(datashape.float32))
        self.assertEqual(dshape('float64'), dshape(datashape.float64))
        self.assertEqual(dshape('complex64'), dshape(datashape.complex64))
        self.assertEqual(dshape('complex128'), dshape(datashape.complex128))
        self.assertEqual(dshape('complex64'), dshape('complex[float32]'))
        self.assertEqual(dshape('complex128'), dshape('complex[float64]'))
        self.assertEqual(dshape("string"), dshape(datashape.string))
        self.assertEqual(dshape("json"), dshape(datashape.json))
        self.assertEqual(dshape("date"), dshape(datashape.date_))
        self.assertEqual(dshape("time"), dshape(datashape.time_))
        self.assertEqual(dshape("datetime"), dshape(datashape.datetime_))

    def test_atom_shape_errors(self):
        self.assertRaises(error.DataShapeSyntaxError, dshape, 'boot')
        self.assertRaises(error.DataShapeSyntaxError, dshape, 'int33')
        self.assertRaises(error.DataShapeSyntaxError, dshape, '12')
        self.assertRaises(error.DataShapeSyntaxError, dshape, 'var')

    @pytest.mark.xfail(reason='implements has not been implemented in the new parser')
    def test_constraints_error(self):
        self.assertRaises(error.DataShapeTypeError, dshape,
                          'A : integral * B : numeric')

    def test_ellipsis_error(self):
        self.assertRaises(error.DataShapeSyntaxError, dshape, 'T * ...')
        self.assertRaises(error.DataShapeSyntaxError, dshape, 'T * S...')

    @pytest.mark.xfail(reason='type decl has been removed in the new parser')
    def test_type_decl(self):
        self.assertRaises(error.DataShapeTypeError, dshape, 'type X T = 3, T')

    @pytest.mark.xfail(reason='type decl has been removed in the new parser')
    def test_type_decl_concrete(self):
        self.assertEqual(dshape('3, int32'), dshape('type X = 3, int32'))

    def test_string_atom(self):
        self.assertEqual(dshape('string'), dshape("string['U8']"))
        self.assertEqual(dshape("string['ascii']")[0].encoding, 'A')
        self.assertEqual(dshape("string['A']")[0].encoding, 'A')
        self.assertEqual(dshape("string['utf-8']")[0].encoding, 'U8')
        self.assertEqual(dshape("string['U8']")[0].encoding, 'U8')
        self.assertEqual(dshape("string['utf-16']")[0].encoding, 'U16')
        self.assertEqual(dshape("string['U16']")[0].encoding, 'U16')
        self.assertEqual(dshape("string['utf-32']")[0].encoding, 'U32')
        self.assertEqual(dshape("string['U32']")[0].encoding, 'U32')

    def test_time(self):
        self.assertEqual(dshape('time')[0].tz, None)
        self.assertEqual(dshape('time[tz="UTC"]')[0].tz, 'UTC')
        self.assertEqual(dshape('time[tz="America/Vancouver"]')[0].tz,
                         'America/Vancouver')
        self.assertEqual(str(dshape('time[tz="UTC"]')), "time[tz='UTC']")

    def test_datetime(self):
        self.assertEqual(dshape('datetime')[0].tz, None)
        self.assertEqual(dshape('datetime[tz="UTC"]')[0].tz, 'UTC')
        self.assertEqual(dshape('datetime[tz="America/Vancouver"]')[0].tz,
                         'America/Vancouver')
        self.assertEqual(str(dshape('datetime[tz="UTC"]')),
                         "datetime[tz='UTC']")

    def test_units(self):
        self.assertEqual(dshape('units["second"]')[0].unit, 'second')
        self.assertEqual(dshape('units["second"]')[0].tp, dshape('float64'))
        self.assertEqual(dshape('units["second", int32]')[0].unit, 'second')
        self.assertEqual(dshape('units["second", int32]')[0].tp,
                         dshape('int32'))

    def test_empty_struct(self):
        self.assertEqual(dshape('{}'), DataShape(Record([])))

    def test_struct_of_array(self):
        self.assertEqual(str(dshape('5 * int32')), '5 * int32')
        self.assertEqual(str(dshape('{field: 5 * int32}')),
                         '{field: 5 * int32}')
        self.assertEqual(str(dshape('{field: M * int32}')),
                         '{field: M * int32}')

    def test_ragged_array(self):
        self.assertTrue(isinstance(dshape('3 * var * int32')[1],
                        datashape.Var))

    def test_from_numpy_fields(self):
        import numpy as np
        dt = np.dtype('i4,i8,f8')
        ds = datashape.from_numpy((), dt)
        self.assertEqual(ds.names, ['f0', 'f1', 'f2'])
        self.assertEqual(ds.types,
                         [datashape.int32, datashape.int64, datashape.float64])

    def test_to_numpy_fields(self):
        import numpy as np
        ds = datashape.dshape('{x: int32, y: float32}')
        shape, dt = datashape.to_numpy(ds)
        self.assertEqual(shape, ())
        self.assertEqual(dt, np.dtype([('x', 'int32'), ('y', 'float32')]))

    def test_syntax(self):
        self.assertEqual(datashape.Fixed(3) * dshape('int32'),
                         dshape('3 * int32'))
        self.assertEqual(3 * dshape('int32'),
                         dshape('3 * int32'))
        self.assertEqual(datashape.Var() * dshape('int32'),
                         dshape('var * int32'))
        self.assertEqual(datashape.Var() * datashape.int32,
                         dshape('var * int32'))
        self.assertEqual(datashape.Var() * 'int32',
                         dshape('var * int32'))
        self.assertEqual(3 * datashape.int32,
                         dshape('3 * int32'))

    def test_python_containers(self):
        var = datashape.Var()
        int32 = datashape.int32
        self.assertEqual(dshape('3 * int32'),
                         dshape((3, int32)))
        self.assertEqual(dshape('3 * int32'),
                         dshape([3, int32]))
        self.assertEqual(dshape('var * 3 * int32'),
                         dshape((var, 3, int32)))

    dshapes = ['bool',
               'int8',
               'int16',
               'int32',
               'int64',
               'uint8',
               'uint16',
               'uint32',
               'uint64',
               'float32',
               'float64',
               'complex64',
               'complex128',
               'string',
               'json',
               'date',
               'time',
               'datetime',
               'int',
               'real',
               'complex',
               'intptr',
               'uintptr',
               '{id: int8, value: bool, result: int16}',
               '{a: int32, b: int64, x: uint8, y: uint16, z: uint32}',
               '{a: float32, b: float64, c: complex64, d: complex128, '
               ' e: string, f: json, g: date, h: time, i: datetime}']

    dimensions = ['2',
                  '100',
                  '...',
                  'var',
                  '2 * var * 2',
                  ]

    def test_dshape_into_repr(self):
        for ds in self.dshapes:
            self.assertEqual(eval(repr(dshape(ds))), dshape(ds))
            for dm in self.dimensions:
                d = dshape(dm + ' * ' + ds)
                self.assertEqual(eval(repr(d)), d)


pointer_sizes = {
    4: {
        'intptr': datashape.int32,
        'uintptr': datashape.uint32,
    },
    8: {
        'intptr': datashape.int64,
        'uintptr': datashape.uint64,
    }
}


@pytest.mark.parametrize('kind', ['intptr', 'uintptr'])
def test_intptr_size(kind):
    assert (dshape(kind) ==
            dshape(pointer_sizes[ctypes.sizeof(ctypes.c_void_p)][kind]))
