import numpy as np

from .util import collect, dshape
from .internal_utils import remove
from .coretypes import (DataShape, Fixed, Var, Ellipsis, Record, Tuple, Unit,
                        date_, datetime_, TypeVar, to_numpy_dtype, Map,
                        Option, Categorical)
from .typesets import floating, boolean

# https://github.com/blaze/datashape/blob/master/docs/source/types.rst

__all__ = ['isdimension', 'ishomogeneous', 'istabular', 'isfixed', 'isscalar',
           'isrecord', 'iscollection', 'isnumeric', 'isboolean', 'isdatelike',
           'isreal']

dimension_types = Fixed, Var, Ellipsis, int


def isscalar(ds):
    """ Is this dshape a single dtype?

    >>> isscalar('int')
    True
    >>> isscalar('?int')
    True
    >>> isscalar('{name: string, amount: int}')
    False
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape) and len(ds) == 1:
        ds = ds[0]
    return isinstance(getattr(ds, 'ty', ds), (Unit, Categorical))


def isrecord(ds):
    """ Is this dshape a record type?

    >>> isrecord('{name: string, amount: int}')
    True
    >>> isrecord('int')
    False
    >>> isrecord('?{name: string, amount: int}')
    True
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape) and len(ds) == 1:
        ds = ds[0]
    return isinstance(getattr(ds, 'ty', ds), Record)


def isdimension(ds):
    """ Is a component a dimension?

    >>> from datashape import int32
    >>> isdimension(Fixed(10))
    True
    >>> isdimension(Var())
    True
    >>> isdimension(int32)
    False
    """
    return isinstance(ds, dimension_types)


def ishomogeneous(ds):
    """ Does datashape contain only one dtype?

    >>> from datashape import int32
    >>> ishomogeneous(int32)
    True
    >>> ishomogeneous('var * 3 * string')
    True
    >>> ishomogeneous('var * {name: string, amount: int}')
    False
    """
    ds = dshape(ds)
    return len(set(remove(isdimension, collect(isscalar, ds)))) == 1


def _dimensions(ds):
    """Number of dimensions of datashape
    """
    return len(dshape(ds).shape)


def isfixed(ds):
    """ Contains no variable dimensions

    >>> isfixed('10 * int')
    True
    >>> isfixed('var * int')
    False
    >>> isfixed('10 * {name: string, amount: int}')
    True
    >>> isfixed('10 * {name: string, amounts: var * int}')
    False
    """
    ds = dshape(ds)
    if isinstance(ds[0], TypeVar):
        return None  # don't know
    if isinstance(ds[0], Var):
        return False
    if isinstance(ds[0], Record):
        return all(map(isfixed, ds[0].types))
    if len(ds) > 1:
        return isfixed(ds.subarray(1))
    return True


def istabular(ds):
    """ A collection of records

    >>> istabular('var * {name: string, amount: int}')
    True
    >>> istabular('var * 10 * 3 * int')
    False
    >>> istabular('10 * var * int')
    False
    >>> istabular('var * (int64, string, ?float64)')
    False
    """
    ds = dshape(ds)
    return _dimensions(ds) == 1 and isrecord(ds.measure)


def iscollection(ds):
    """ Is a collection of items, has dimension

    >>> iscollection('5 * int32')
    True
    >>> iscollection('int32')
    False
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    return isdimension(ds[0])


def isnumeric(ds):
    """ Has a numeric measure

    >>> isnumeric('int32')
    True
    >>> isnumeric('3 * ?real')
    True
    >>> isnumeric('string')
    False
    >>> isnumeric('var * {amount: ?int32}')
    False
    """
    ds = launder(ds)

    try:
        npdtype = to_numpy_dtype(ds)
    except TypeError:
        return False
    else:
        return isinstance(ds, Unit) and np.issubdtype(npdtype, np.number)


def launder(ds):
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape):
        ds = ds.measure
    return getattr(ds, 'ty', ds)


def isreal(ds):
    """ Has a numeric measure

    >>> isreal('float32')
    True
    >>> isreal('3 * ?real')
    True
    >>> isreal('string')
    False
    """
    ds = launder(ds)
    return isinstance(ds, Unit) and ds in floating


def isboolean(ds):
    """ Has a boolean measure

    >>> isboolean('bool')
    True
    >>> isboolean('3 * ?bool')
    True
    >>> isboolean('int')
    False
    """
    return launder(ds) in boolean


def isdatelike(ds):
    """ Has a date or datetime measure

    >>> isdatelike('int32')
    False
    >>> isdatelike('3 * datetime')
    True
    >>> isdatelike('?datetime')
    True
    """
    ds = launder(ds)
    return ds == date_ or ds == datetime_
