"""
Test the DataShape parser.
"""

from __future__ import absolute_import, division, print_function

import unittest
import pytest

import datashape
from datashape.util.testing import assert_dshape_equal
from datashape.parser import parse
from datashape import coretypes as ct
from datashape import DataShapeSyntaxError


@pytest.fixture
def sym():
    return datashape.TypeSymbolTable()


class TestDataShapeParseBasicDType(unittest.TestCase):

    def setUp(self):
        # Create a default symbol table for the parser to use
        self.sym = datashape.TypeSymbolTable()

    def test_bool(self):
        self.assertEqual(parse('bool', self.sym),
                         ct.DataShape(ct.bool_))

    def test_signed_integers(self):
        self.assertEqual(parse('int8', self.sym),
                         ct.DataShape(ct.int8))
        self.assertEqual(parse('int16', self.sym),
                         ct.DataShape(ct.int16))
        self.assertEqual(parse('int32', self.sym),
                         ct.DataShape(ct.int32))
        self.assertEqual(parse('int64', self.sym),
                         ct.DataShape(ct.int64))
        # self.assertEqual(parse('int128', self.sym),
        #                 ct.DataShape(ct.int128))
        self.assertEqual(parse('int', self.sym),
                         ct.DataShape(ct.int_))
        # 'int' is an alias for 'int32'
        self.assertEqual(parse('int', self.sym),
                         parse('int32', self.sym))
        self.assertEqual(parse('intptr', self.sym),
                         ct.DataShape(ct.intptr))

    def test_unsigned_integers(self):
        self.assertEqual(parse('uint8', self.sym),
                         ct.DataShape(ct.uint8))
        self.assertEqual(parse('uint16', self.sym),
                         ct.DataShape(ct.uint16))
        self.assertEqual(parse('uint32', self.sym),
                         ct.DataShape(ct.uint32))
        self.assertEqual(parse('uint64', self.sym),
                         ct.DataShape(ct.uint64))
        # self.assertEqual(parse('uint128', self.sym),
        #                 ct.DataShape(ct.uint128))
        self.assertEqual(parse('uintptr', self.sym),
                         ct.DataShape(ct.uintptr))

    def test_float(self):
        self.assertEqual(parse('float16', self.sym),
                         ct.DataShape(ct.float16))
        self.assertEqual(parse('float32', self.sym),
                         ct.DataShape(ct.float32))
        self.assertEqual(parse('float64', self.sym),
                         ct.DataShape(ct.float64))
        # self.assertEqual(parse('float128', self.sym),
        #                 ct.DataShape(ct.float128))
        self.assertEqual(parse('real', self.sym),
                         ct.DataShape(ct.real))
        # 'real' is an alias for 'float64'
        self.assertEqual(parse('real', self.sym),
                         parse('float64', self.sym))

    def test_null(self):
        self.assertEqual(parse('null', self.sym), ct.DataShape(ct.null))

    def test_void(self):
        self.assertEqual(parse('void', self.sym), ct.DataShape(ct.void))

    def test_object(self):
        self.assertEqual(parse('object', self.sym), ct.DataShape(ct.object_))

    def test_complex(self):
        self.assertEqual(parse('complex[float32]', self.sym),
                         ct.DataShape(ct.complex_float32))
        self.assertEqual(parse('complex[float64]', self.sym),
                         ct.DataShape(ct.complex_float64))
        self.assertEqual(parse('complex', self.sym),
                         ct.DataShape(ct.complex_))
        # 'complex' is an alias for 'complex[float64]'
        self.assertEqual(parse('complex', self.sym),
                         parse('complex[float64]', self.sym))

    def test_option(self):
        self.assertEqual(parse('option[int32]', self.sym),
                         ct.DataShape(ct.Option(ct.int32)))
        self.assertEqual(parse('?int32', self.sym),
                         ct.DataShape(ct.Option(ct.int32)))
        self.assertEqual(parse('2 * 3 * option[int32]', self.sym),
                         ct.DataShape(ct.Fixed(2), ct.Fixed(3),
                                      ct.Option(ct.int32)))
        self.assertEqual(parse('2 * 3 * ?int32', self.sym),
                         ct.DataShape(ct.Fixed(2), ct.Fixed(3),
                                      ct.Option(ct.int32)))
        self.assertEqual(parse('2 * option[3 * int32]', self.sym),
                         ct.DataShape(ct.Fixed(2),
                                      ct.Option(ct.DataShape(ct.Fixed(3),
                                                             ct.int32))))
        self.assertEqual(parse('2 * ?3 * int32', self.sym),
                         ct.DataShape(ct.Fixed(2),
                                      ct.Option(ct.DataShape(ct.Fixed(3),
                                                             ct.int32))))

    def test_raise(self):
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse, '', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse, 'boot', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse, 'int33', self.sym)


class TestDataShapeParserDTypeConstr(unittest.TestCase):

    def test_unary_dtype_constr(self):
        # Create a symbol table with no types in it, so we can
        # make some isolated type constructors for testing
        sym = datashape.TypeSymbolTable(bare=True)
        # A limited set of dtypes for testing
        sym.dtype['int8'] = ct.int8
        sym.dtype['uint16'] = ct.uint16
        sym.dtype['float64'] = ct.float64
        # TypeVar type constructor
        sym.dtype_constr['typevar'] = ct.TypeVar
        # Unary dtype constructor that asserts on the argument value
        expected_blah = [None]

        def _unary_type_constr(blah):
            self.assertEqual(blah, expected_blah[0])
            expected_blah[0] = None
            return ct.float32
        sym.dtype_constr['unary'] = _unary_type_constr

        def assertExpectedParse(ds_str, expected):
            # Set the expected value, and call the parser
            expected_blah[0] = expected
            self.assertEqual(parse(ds_str, sym), ct.DataShape(ct.float32))
            # Make sure the expected value was actually run by
            # check that it reset the expected value to None
            self.assertEqual(expected_blah[0], None,
                             'The test unary type constructor did not run')

        # Integer parameter (positional)
        assertExpectedParse('unary[0]', 0)
        assertExpectedParse('unary[100000]', 100000)
        # String parameter (positional)
        assertExpectedParse('unary["test"]', 'test')
        assertExpectedParse("unary['test']", 'test')
        assertExpectedParse('unary["\\uc548\\ub155"]', u'\uc548\ub155')
        assertExpectedParse(u'unary["\uc548\ub155"]', u'\uc548\ub155')
        # DataShape parameter (positional)
        assertExpectedParse('unary[int8]', ct.DataShape(ct.int8))
        assertExpectedParse('unary[X]', ct.DataShape(ct.TypeVar('X')))
        # Empty list parameter (positional)
        assertExpectedParse('unary[[]]', [])
        # List of integers parameter (positional)
        assertExpectedParse('unary[[0, 3, 12]]', [0, 3, 12])
        # List of strings parameter (positional)
        assertExpectedParse('unary[["test", "one", "two"]]',
                            ["test", "one", "two"])
        # List of datashapes parameter (positional)
        assertExpectedParse('unary[[float64, int8, uint16]]',
                            [ct.DataShape(ct.float64), ct.DataShape(ct.int8),
                             ct.DataShape(ct.uint16)])

        # Integer parameter (keyword)
        assertExpectedParse('unary[blah=0]', 0)
        assertExpectedParse('unary[blah=100000]', 100000)
        # String parameter (keyword)
        assertExpectedParse('unary[blah="test"]', 'test')
        assertExpectedParse("unary[blah='test']", 'test')
        assertExpectedParse('unary[blah="\\uc548\\ub155"]', u'\uc548\ub155')
        assertExpectedParse(u'unary[blah="\uc548\ub155"]', u'\uc548\ub155')
        # DataShape parameter (keyword)
        assertExpectedParse('unary[blah=int8]', ct.DataShape(ct.int8))
        assertExpectedParse('unary[blah=X]', ct.DataShape(ct.TypeVar('X')))
        # Empty list parameter (keyword)
        assertExpectedParse('unary[blah=[]]', [])
        # List of integers parameter (keyword)
        assertExpectedParse('unary[blah=[0, 3, 12]]', [0, 3, 12])
        # List of strings parameter (keyword)
        assertExpectedParse('unary[blah=["test", "one", "two"]]',
                            ["test", "one", "two"])
        # List of datashapes parameter (keyword)
        assertExpectedParse('unary[blah=[float64, int8, uint16]]',
                            [ct.DataShape(ct.float64), ct.DataShape(ct.int8),
                             ct.DataShape(ct.uint16)])

    def test_binary_dtype_constr(self):
        # Create a symbol table with no types in it, so we can
        # make some isolated type constructors for testing
        sym = datashape.TypeSymbolTable(bare=True)
        # A limited set of dtypes for testing
        sym.dtype['int8'] = ct.int8
        sym.dtype['uint16'] = ct.uint16
        sym.dtype['float64'] = ct.float64
        # TypeVar type constructor
        sym.dtype_constr['typevar'] = ct.TypeVar
        # Binary dtype constructor that asserts on the argument values
        expected_arg = [None, None]

        def _binary_type_constr(a, b):
            self.assertEqual(a, expected_arg[0])
            self.assertEqual(b, expected_arg[1])
            expected_arg[0] = None
            expected_arg[1] = None
            return ct.float32
        sym.dtype_constr['binary'] = _binary_type_constr

        def assertExpectedParse(ds_str, expected_a, expected_b):
            # Set the expected value, and call the parser
            expected_arg[0] = expected_a
            expected_arg[1] = expected_b
            self.assertEqual(parse(ds_str, sym), ct.DataShape(ct.float32))
            # Make sure the expected value was actually run by
            # check that it reset the expected value to None
            self.assertEqual(expected_arg, [None, None],
                             'The test binary type constructor did not run')

        # Positional args
        assertExpectedParse('binary[1, 0]', 1, 0)
        assertExpectedParse('binary[0, "test"]', 0, 'test')
        assertExpectedParse('binary[int8, "test"]',
                            ct.DataShape(ct.int8), 'test')
        assertExpectedParse('binary[[1,3,5], "test"]', [1, 3, 5], 'test')
        # Positional and keyword args
        assertExpectedParse('binary[0, b=1]', 0, 1)
        assertExpectedParse('binary["test", b=A]', 'test',
                            ct.DataShape(ct.TypeVar('A')))
        assertExpectedParse('binary[[3, 6], b=int8]', [3, 6],
                            ct.DataShape(ct.int8))
        assertExpectedParse('binary[Arg, b=["x", "test"]]',
                            ct.DataShape(ct.TypeVar('Arg')), ['x', 'test'])
        # Keyword args
        assertExpectedParse('binary[a=1, b=0]', 1, 0)
        assertExpectedParse('binary[a=[int8, A, uint16], b="x"]',
                            [ct.DataShape(ct.int8),
                             ct.DataShape(ct.TypeVar('A')),
                             ct.DataShape(ct.uint16)],
                            'x')

    def test_dtype_constr_errors(self):
        # Create a symbol table with no types in it, so we can
        # make some isolated type constructors for testing
        sym = datashape.TypeSymbolTable(bare=True)
        # A limited set of dtypes for testing
        sym.dtype['int8'] = ct.int8
        sym.dtype['uint16'] = ct.uint16
        sym.dtype['float64'] = ct.float64
        # Arbitrary dtype constructor that does nothing

        def _type_constr(*args, **kwargs):
            return ct.float32
        sym.dtype_constr['tcon'] = _type_constr

        # Require closing "]"
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[', sym)
        # Type constructors should always have an argument
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[]', sym)
        # Unknown type
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[unknown]', sym)
        # Missing parameter value
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[x=', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[x=]', sym)
        # A positional arg cannot be after a keyword arg
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[x=A, B]', sym)
        # List args must be homogeneous
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[[0, "x"]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[[0, X]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[["x", 0]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[["x", X]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[[X, 0]]', sym)
        self.assertRaises(DataShapeSyntaxError,
                          parse, 'tcon[[X, "x"]]', sym)


class TestDataShapeParserDims(unittest.TestCase):

    def setUp(self):
        # Create a default symbol table for the parser to use
        self.sym = datashape.TypeSymbolTable()

    def test_fixed_dims(self):
        self.assertEqual(parse('3 * bool', self.sym),
                         ct.DataShape(ct.Fixed(3), ct.bool_))
        self.assertEqual(parse('7 * 3 * bool', self.sym),
                         ct.DataShape(ct.Fixed(7), ct.Fixed(3), ct.bool_))
        self.assertEqual(parse('5 * 3 * 12 * bool', self.sym),
                         ct.DataShape(ct.Fixed(5), ct.Fixed(3),
                                      ct.Fixed(12), ct.bool_))
        self.assertEqual(parse('2 * 3 * 4 * 5 * bool', self.sym),
                         ct.DataShape(ct.Fixed(2), ct.Fixed(3),
                                      ct.Fixed(4), ct.Fixed(5), ct.bool_))

    def test_typevar_dims(self):
        self.assertEqual(parse('M * bool', self.sym),
                         ct.DataShape(ct.TypeVar('M'), ct.bool_))
        self.assertEqual(parse('A * B * bool', self.sym),
                         ct.DataShape(ct.TypeVar('A'), ct.TypeVar('B'), ct.bool_))
        self.assertEqual(parse('A... * X * 3 * bool', self.sym),
                         ct.DataShape(ct.Ellipsis(ct.TypeVar('A')), ct.TypeVar('X'),
                                      ct.Fixed(3), ct.bool_))

    def test_var_dims(self):
        self.assertEqual(parse('var * bool', self.sym),
                         ct.DataShape(ct.Var(), ct.bool_))
        self.assertEqual(parse('var * var * bool', self.sym),
                         ct.DataShape(ct.Var(), ct.Var(), ct.bool_))
        self.assertEqual(parse('M * 5 * var * bool', self.sym),
                         ct.DataShape(ct.TypeVar('M'), ct.Fixed(5), ct.Var(), ct.bool_))

    def test_ellipses(self):
        self.assertEqual(parse('... * bool', self.sym),
                         ct.DataShape(ct.Ellipsis(), ct.bool_))
        self.assertEqual(parse('M * ... * bool', self.sym),
                         ct.DataShape(ct.TypeVar('M'), ct.Ellipsis(), ct.bool_))
        self.assertEqual(parse('M * ... * 3 * bool', self.sym),
                         ct.DataShape(ct.TypeVar('M'), ct.Ellipsis(),
                                      ct.Fixed(3), ct.bool_))


class TestDataShapeParseStruct(unittest.TestCase):

    def setUp(self):
        # Create a default symbol table for the parser to use
        self.sym = datashape.TypeSymbolTable()

    def test_struct(self):
        # Simple struct
        self.assertEqual(parse('{x: int16, y: int32}', self.sym),
                         ct.DataShape(ct.Record([('x', ct.DataShape(ct.int16)),
                                                 ('y', ct.DataShape(ct.int32))])))
        # A trailing comma is ok
        self.assertEqual(parse('{x: int16, y: int32,}', self.sym),
                         ct.DataShape(ct.Record([('x', ct.DataShape(ct.int16)),
                                                 ('y', ct.DataShape(ct.int32))])))
        # Field names starting with _ and caps
        self.assertEqual(parse('{_x: int16, Zed: int32,}', self.sym),
                         ct.DataShape(ct.Record([('_x', ct.DataShape(ct.int16)),
                                                 ('Zed', ct.DataShape(ct.int32))])))
        # A slightly bigger example
        ds_str = """3 * var * {
                        id : int32,
                        name : string,
                        description : {
                            language : string,
                            text : string
                        },
                        entries : var * {
                            date : date,
                            text : string
                        }
                    }"""
        int32 = ct.DataShape(ct.int32)
        string = ct.DataShape(ct.string)
        date = ct.DataShape(ct.date_)
        ds = (ct.Fixed(3), ct.Var(),
              ct.Record([('id', int32),
                         ('name', string),
                         ('description', ct.DataShape(ct.Record([('language', string),
                                                                 ('text', string)]))),
                         ('entries', ct.DataShape(ct.Var(),
                                                  ct.Record([('date', date),
                                                             ('text', string)])))]))
        self.assertEqual(parse(ds_str, self.sym), ct.DataShape(*ds))

    def test_fields_with_dshape_names(self):
        # Should be able to name a field 'type', 'int64', etc
        ds = parse("""{
                type: bool,
                data: bool,
                blob: bool,
                bool: bool,
                int: int32,
                float: float32,
                double: float64,
                int8: int8,
                int16: int16,
                int32: int32,
                int64: int64,
                uint8: uint8,
                uint16: uint16,
                uint32: uint32,
                uint64: uint64,
                float16: float32,
                float32: float32,
                float64: float64,
                float128: float64,
                complex: float32,
                complex64: float32,
                complex128: float64,
                string: string,
                object: string,
                datetime: string,
                datetime64: string,
                timedelta: string,
                timedelta64: string,
                json: string,
                var: string,
            }""", self.sym)
        self.assertEqual(type(ds[-1]), ct.Record)
        self.assertEqual(len(ds[-1].names), 30)

    def test_kiva_datashape(self):
        # A slightly more complicated datashape which should parse
        ds = parse("""5 * var * {
              id: int64,
              name: string,
              description: {
                languages: var * string[2],
                texts: json,
              },
              status: string,
              funded_amount: float64,
              basket_amount: json,
              paid_amount: json,
              image: {
                id: int64,
                template_id: int64,
              },
              video: json,
              activity: string,
              sector: string,
              use: string,
              delinquent: bool,
              location: {
                country_code: string[2],
                country: string,
                town: json,
                geo: {
                  level: string,
                  pairs: string,
                  type: string,
                },
              },
              partner_id: int64,
              posted_date: json,
              planned_expiration_date: json,
              loan_amount: float64,
              currency_exchange_loss_amount: json,
              borrowers: var * {
                first_name: string,
                last_name: string,
                gender: string[1],
                pictured: bool,
              },
              terms: {
                disbursal_date: json,
                disbursal_currency: string[3,'A'],
                disbursal_amount: float64,
                loan_amount: float64,
                local_payments: var * {
                  due_date: json,
                  amount: float64,
                },
                scheduled_payments: var * {
                  due_date: json,
                  amount: float64,
                },
                loss_liability: {
                  nonpayment: string,
                  currency_exchange: string,
                  currency_exchange_coverage_rate: json,
                },
              },
              payments: var * {
                amount: float64,
                local_amount: float64,
                processed_date: json,
                settlement_date: json,
                rounded_local_amount: float64,
                currency_exchange_loss_amount: float64,
                payment_id: int64,
                comment: json,
              },
              funded_date: json,
              paid_date: json,
              journal_totals: {
                entries: int64,
                bulkEntries: int64,
              },
            }
        """, self.sym)
        self.assertEqual(type(ds[-1]), ct.Record)
        self.assertEqual(len(ds[-1].names), 25)

    def test_strings_in_ds(self):
        # Name the fields with some arbitrary string!
        ds = parse("""5 * var * {
              id: int64,
             'my field': string,
              name: string }
             """, self.sym)
        self.assertEqual(len(ds[-1].names), 3)
        ds = parse("""2 * var * {
             "AASD @#$@#$ \' sdf": string,
              id: float32,
              id2: int64,
              name: string }
             """, self.sym)
        self.assertEqual(len(ds[-1].names), 4)

    def test_struct_errors(self):
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse,
                          '{id: int64, name: string amount: invalidtype}',
                          self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse,
                          '{id: int64, name: string, amount: invalidtype}',
                          self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse,
                          '{id: int64, name: string, amount: %}',
                          self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse,
                          "{\n" +
                          "   id: int64;\n" +
                          "   name: string;\n" +
                          "   amount+ float32;\n" +
                          "}\n",
                          self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse,
                          "{\n" +
                          "   id: int64;\n" +
                          "   'my field 1': string;\n" +
                          "   amount+ float32;\n" +
                          "}\n",
                          self.sym)
        # Don't accept explicitly Unicode string literals
        self.assertRaises(datashape.DataShapeSyntaxError,
                          parse,
                          "{\n" +
                          "   id: int64,\n" +
                          "   u'my field 1': string,\n" +
                          "   amount: float32\n" +
                          "}\n",
                          self.sym)


class TestDataShapeParseTuple(unittest.TestCase):

    def setUp(self):
        # Create a default symbol table for the parser to use
        self.sym = datashape.TypeSymbolTable()

    def test_tuple(self):
        # Simple tuple
        self.assertEqual(parse('(float32)', self.sym),
                         ct.DataShape(ct.Tuple([ct.DataShape(ct.float32)])))
        self.assertEqual(parse('(int16, int32)', self.sym),
                         ct.DataShape(ct.Tuple([ct.DataShape(ct.int16),
                                                ct.DataShape(ct.int32)])))
        # A trailing comma is ok
        self.assertEqual(parse('(float32,)', self.sym),
                         ct.DataShape(ct.Tuple([ct.DataShape(ct.float32)])))
        self.assertEqual(parse('(int16, int32,)', self.sym),
                         ct.DataShape(ct.Tuple([ct.DataShape(ct.int16),
                                                ct.DataShape(ct.int32)])))


def test_funcproto(sym):
    # Simple funcproto
    assert (parse('(float32) -> float64', sym) ==
            ct.DataShape(ct.Function(ct.DataShape(ct.float32),
                                     ct.DataShape(ct.float64))))
    assert (parse('(int16, int32) -> bool', sym) ==
            ct.DataShape(ct.Function(ct.DataShape(ct.int16),
                                     ct.DataShape(ct.int32),
                                     ct.DataShape(ct.bool_))))
    # A trailing comma is ok
    assert (parse('(float32,) -> float64', sym) ==
            ct.DataShape(ct.Function(ct.DataShape(ct.float32),
                                     ct.DataShape(ct.float64))))
    assert_dshape_equal(
        parse('(int16, int32,) -> bool', sym),
        ct.DataShape(ct.Function(
            ct.DataShape(ct.int16),
            ct.DataShape(ct.int32),
            ct.DataShape(ct.bool_)
        ))
    )

    # Empty argument signature.
    assert_dshape_equal(
        parse('() -> bool', sym),
        ct.DataShape(ct.Function(
            ct.DataShape(ct.bool_),
        ))
    )


def test_funcproto_no_return_type(sym):
    with pytest.raises(DataShapeSyntaxError):
        parse('(int64, int32) ->', sym)


def test_empty_tuple(sym):
    t = parse('()', sym)
    assert isinstance(t, ct.DataShape)
    assert isinstance(t.measure, ct.Tuple)
    assert t.measure.dshapes == ()


def test_no_right_paren_tuple(sym):
    with pytest.raises(DataShapeSyntaxError):
        parse('(int64', sym)


def test_garbage_at_end(sym):
    with pytest.raises(DataShapeSyntaxError):
        parse('int64,asdf', sym)


def test_type_constructor_fail(sym):
    with pytest.raises(DataShapeSyntaxError):
        parse('string[10,[', sym)

    with pytest.raises(DataShapeSyntaxError):
        parse('string[10,', sym)


def test_dim_constructor_fail(sym):
    with pytest.raises(NotImplementedError):
        parse('fixed[10] * var * string', sym)
    with pytest.raises(DataShapeSyntaxError):
        parse('fixed[10 * var * string', sym)


def test_invalid_dtype(sym):
    with pytest.raises(DataShapeSyntaxError):
        parse('10 * foo[10]', sym)
