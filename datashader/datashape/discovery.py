from __future__ import print_function, division, absolute_import

from collections import OrderedDict
from datetime import datetime, date, time, timedelta
from itertools import chain
import re
from textwrap import dedent
from types import MappingProxyType
from warnings import warn

from dateutil.parser import parse as dateparse
import numpy as np

from .dispatch import dispatch
from .coretypes import (int32, int64, float64, bool_, complex128, datetime_,
                        Option, var, from_numpy, Tuple, null,
                        Record, string, Null, DataShape, real, date_, time_,
                        Unit, timedelta_, TimeDelta, object_, String)
from .predicates import isdimension, isrecord
from .internal_utils import _toposort, groupby
from .util import subclasses


__all__ = ['discover']


@dispatch(object)
def discover(obj, **kwargs):
    """ Discover datashape of object

    A datashape encodes the datatypes and the shape/length of an object.
    Discover returns the datashape of a Python object.  This object can refer
    to external data.

    Datashapes range from simple scalars

    >>> discover(10)
    ctype('int64')

    To collections

    >>> discover([[1, 2, 3], [4, 5, 6]])
    dshape('2 * 3 * int64')

    To record types and other objects

    >>> x = np.array([('Alice', 100), ('Bob', 200)], dtype=[('name', 'S7'),
    ...                                                     ('amount', 'i4')])
    >>> discover(x)
    dshape('2 * {name: string[7, "ascii"], amount: int32}')

    See http://datashape.pydata.org/grammar.html#some-simple-examples
    for more examples
    """
    type_name = type(obj).__name__
    if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
        warn(
            dedent(
                """\
                array-like discovery is deprecated.
                Please write an explicit discover function for type '%s'.
                """ % type_name,
            ),
            DeprecationWarning,
        )
        return from_numpy(obj.shape, obj.dtype)
    raise NotImplementedError("Don't know how to discover type %r" % type_name)


@dispatch(int)
def discover(i):  # noqa: F811
    return int64


npinttypes = tuple(chain.from_iterable((x for x in subclasses(icls)
                                        if x.__name__.startswith(('int',
                                                                  'uint')))
                                       for icls in subclasses(np.integer)))


@dispatch(bytes)
def discover(b):  # noqa: F811
    return String('A')


@dispatch(npinttypes)
def discover(n):  # noqa: F811
    return from_numpy((), n.dtype)


@dispatch(float)
def discover(f):  # noqa: F811
    return float64


@dispatch(bool)
def discover(b):  # noqa: F811
    return bool_


@dispatch(complex)
def discover(z):  # noqa: F811
    return complex128


@dispatch(datetime)
def discover(dt):  # noqa: F811
    return datetime_


@dispatch(timedelta)
def discover(td):  # noqa: F811
    return TimeDelta(unit='us')


@dispatch(date)
def discover(dt):  # noqa: F811
    return date_


@dispatch(time)
def discover(t):  # noqa: F811
    return time_


@dispatch((type(None), Null))
def discover(i):  # noqa: F811
    return null


bools = {'False': False,
         'false': False,
         'True': True,
         'true': True}


def timeparse(x, formats=('%H:%M:%S', '%H:%M:%S.%f')):
    msg = ''
    for format in formats:
        try:
            return datetime.strptime(x, format).time()
        except ValueError as e:  # raises if it doesn't match the format
            msg = str(e)
    raise ValueError(msg)


def deltaparse(x):
    """Naive timedelta string parser

    Examples
    --------
    >>> td = '1 day'
    >>> deltaparse(td)  # doctest: +SKIP
    numpy.timedelta64(1,'D')
    >>> deltaparse('1.2 days')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: floating point timedelta value not supported
    """
    value, unit = re.split(r'\s+', x.strip())
    value = float(value)
    if not value.is_integer():
        raise ValueError('floating point timedelta values not supported')
    return np.timedelta64(int(value), TimeDelta(unit=unit).unit)


string_coercions = int, float, bools.__getitem__, deltaparse, timeparse


def is_zero_time(t):
    return not (t.hour or t.minute or t.second or t.microsecond)


@dispatch(str)
def discover(s):  # noqa: F811
    if not s:
        return null

    for f in string_coercions:
        try:
            return discover(f(s))
        except (ValueError, KeyError):
            pass

    # don't let dateutil parse things like sunday, monday etc into dates
    if s.isalpha() or s.isspace():
        return string

    try:
        d = dateparse(s)
    except (ValueError, OverflowError):  # OverflowError for stuff like 'INF...'
        pass
    else:
        return date_ if is_zero_time(d.time()) else datetime_

    return string


@dispatch((tuple, list, set, frozenset))
def discover(seq):  # noqa: F811
    if not seq:
        return var * string
    unite = do_one([unite_identical, unite_base, unite_merge_dimensions])
    # [(a, b), (a, c)]
    if (all(isinstance(item, (tuple, list)) for item in seq) and
            len(set(map(len, seq))) == 1):
        columns = list(zip(*seq))
        try:
            types = [unite([discover(data) for data in column]).subshape[0]
                     for column in columns]
            unite = do_one([unite_identical, unite_merge_dimensions, Tuple])
            return len(seq) * unite(types)
        except AttributeError:  # no subshape available
            pass

    # [{k: v, k: v}, {k: v, k: v}]
    if all(isinstance(item, dict) for item in seq):
        keys = sorted(set.union(*(set(d) for d in seq)))
        columns = [[item.get(key) for item in seq] for key in keys]
        try:
            types = [unite([discover(data) for data in column]).subshape[0]
                     for column in columns]
            return len(seq) * Record(list(zip(keys, types)))
        except AttributeError:
            pass

    types = list(map(discover, seq))
    return do_one([unite_identical, unite_merge_dimensions, Tuple])(types)


def isnull(ds):
    return ds == null or ds == DataShape(null)


def identity(x):
    return x


# (a, b) implies that b can turn into a
edges = [
    (string, int64),  # E.g. int64 can be turned into a string
    (string, real),
    (string, date_),
    (string, datetime_),
    (string, timedelta_),
    (string, bool_),
    (datetime_, date_),
    (int64, int32),
    (real, int64),
    (string, null)]

numeric_edges = [
    (int64, int32),
    (real, int64),
    (string, null)
]


# {a: [b, c]} a is more general than b or c
edges = groupby(lambda x: x[1], edges)
edges = dict((k, set(a for a, b in v)) for k, v in edges.items())
toposorted = _toposort(edges)


def lowest_common_dshape(dshapes):
    """ Find common shared dshape

    >>> lowest_common_dshape([int32, int64, float64])
    ctype("float64")

    >>> lowest_common_dshape([int32, int64])
    ctype("int64")

    >>> lowest_common_dshape([string, int64])
    ctype("string")
    """
    common = set.intersection(*[descendents(edges, ds) for ds in dshapes])
    if common and any(c in toposorted for c in common):
        return min(common, key=toposorted.index)
    raise ValueError("Not all dshapes are known.  Extend edges.")


def unite_base(dshapes):
    """ Performs lowest common dshape and also null aware

    >>> unite_base([float64, float64, int64])
    dshape("3 * float64")

    >>> unite_base([int32, int64, null])
    dshape("3 * ?int64")
    """
    dshapes = [unpack(ds) for ds in dshapes]
    bynull = groupby(isnull, dshapes)
    try:
        good_dshapes = bynull[False]
    except KeyError:
        return len(dshapes) * null
    if all(isinstance(ds, Unit) for ds in good_dshapes):
        base = lowest_common_dshape(good_dshapes)
    elif (all(isinstance(ds, Record) for ds in good_dshapes) and
          ds.names == dshapes[0].names for ds in good_dshapes):
        names = good_dshapes[0].names
        base = Record([[name,
                        unite_base([ds.dict.get(name, null) for ds in good_dshapes]).subshape[0]]
                       for name in names])
    if base:
        if bynull.get(True):
            base = Option(base)
        return len(dshapes) * base


def unite_identical(dshapes):
    """

    >>> unite_identical([int32, int32, int32])
    dshape("3 * int32")
    """
    if len(set(dshapes)) == 1:
        return len(dshapes) * dshapes[0]


def unite_merge_dimensions(dshapes, unite=unite_identical):
    """

    >>> unite_merge_dimensions([10 * string, 10 * string])
    dshape("2 * 10 * string")

    >>> unite_merge_dimensions([10 * string, 20 * string])
    dshape("2 * var * string")
    """
    n = len(dshapes)
    if all(isinstance(ds, DataShape) and isdimension(ds[0]) for ds in dshapes):
        dims = [ds[0] for ds in dshapes]
        base = unite([ds.subshape[0] for ds in dshapes])
        if base:
            if len(set(dims)) == 1:
                return n * (dims[0] * base.subshape[0])
            else:
                return n * (var * base.subshape[0])


def do_one(funcs):
    def f(inp):
        for func in funcs:
            result = func(inp)
            if result:
                return result
        return inp
    return f


def unpack(ds):
    """ Unpack DataShape constructor if unnecessary

    Record packs inputs in DataShape containers.  This unpacks it.

    >>> from datashader.datashape import dshape
    >>> unpack(dshape('string'))
    ctype("string")
    """
    if isinstance(ds, DataShape) and len(ds) == 1:
        return ds[0]
    else:
        return ds


@discover.register(dict)
@discover.register(MappingProxyType)
def _mapping_discover(m):
    return Record((k, discover(m[k])) for k in sorted(m))


@dispatch(OrderedDict)
def discover(od):
    return Record((k, discover(v)) for k, v in od.items())


@dispatch(np.number)
def discover(n):
    return from_numpy((), type(n))


@dispatch(np.timedelta64)
def discover(n):
    return from_numpy((), n)


def is_string_array(x):
    """ Is an array of strings

    >>> is_string_array(np.array(['Hello', 'world'], dtype='O'))
    True
    >>> is_string_array(np.array(['Hello', None], dtype='O'))
    False
    """
    return all(isinstance(i, str) for i in x.flat[:5].tolist())


@dispatch(np.ndarray)
def discover(x):
    ds = from_numpy(x.shape, x.dtype)

    # NumPy uses object dtype both for strings (which we want to call string)
    # and for Python objects (which we want to call object)
    # Lets look at the first few elements and check
    if ds.measure == object_ and is_string_array(x):
        return DataShape(*(ds.shape + (string,)))

    if isrecord(ds.measure) and object_ in ds.measure.types:
        m = Record([[name, string if typ == object_ and is_string_array(x[name])
                     else typ]
                    for name, typ in ds.measure.parameters[0]])
        return DataShape(*(ds.shape + (m,)))
    else:
        return ds


def descendents(d, x):
    """

    >>> d = {3: [2], 2: [1, 0], 5: [6]}
    >>> sorted(descendents(d, 3))
    [0, 1, 2, 3]
    """
    desc = set([x])
    children = d.get(x, set())
    while children:
        children = set.union(*[set(d.get(kid, ())) for kid in desc])
        children -= desc
        desc.update(children)
    return desc


Mock = None
try:
    from unittest.mock import Mock
except ImportError:
    try:
        from mock import Mock
    except ImportError:
        pass

if Mock is not None:
    @dispatch(Mock)
    def discover(m):
        raise NotImplementedError("Don't know how to discover mock objects")
del Mock
