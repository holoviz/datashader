
from itertools import chain
import operator

from .. import parser
from .. import type_symbol_table
from ..validation import validate
from .. import coretypes


__all__ = 'dshape', 'dshapes', 'has_var_dim', 'has_ellipsis', 'cat_dshapes'

subclasses = operator.methodcaller('__subclasses__')

#------------------------------------------------------------------------
# Utility Functions for DataShapes
#------------------------------------------------------------------------

def dshapes(*args):
    """
    Parse a bunch of datashapes all at once.

    >>> a, b = dshapes('3 * int32', '2 * var * float64')
    """
    return [dshape(arg) for arg in args]


def dshape(o):
    """
    Parse a datashape. For a thorough description see
    https://datashape.readthedocs.io/en/latest/

    >>> ds = dshape('2 * int32')
    >>> ds[1]
    ctype("int32")
    """
    if isinstance(o, coretypes.DataShape):
        return o
    if isinstance(o, str):
        ds = parser.parse(o, type_symbol_table.sym)
    elif isinstance(o, (coretypes.CType, coretypes.String,
                        coretypes.Record, coretypes.JSON,
                        coretypes.Date, coretypes.Time, coretypes.DateTime,
                        coretypes.Unit)):
        ds = coretypes.DataShape(o)
    elif isinstance(o, coretypes.Mono):
        ds = o
    elif isinstance(o, (list, tuple)):
        ds = coretypes.DataShape(*o)
    else:
        raise TypeError('Cannot create dshape from object of type %s' % type(o))
    validate(ds)
    return ds


def cat_dshapes(dslist):
    """
    Concatenates a list of dshapes together along
    the first axis. Raises an error if there is
    a mismatch along another axis or the measures
    are different.

    Requires that the leading dimension be a known
    size for all data shapes.
    TODO: Relax this restriction to support
          streaming dimensions.

    >>> cat_dshapes(dshapes('10 * int32', '5 * int32'))
    dshape("15 * int32")
    """
    if len(dslist) == 0:
        raise ValueError('Cannot concatenate an empty list of dshapes')
    elif len(dslist) == 1:
        return dslist[0]

    outer_dim_size = operator.index(dslist[0][0])
    inner_ds = dslist[0][1:]
    for ds in dslist[1:]:
        outer_dim_size += operator.index(ds[0])
        if ds[1:] != inner_ds:
            raise ValueError(('The datashapes to concatenate much'
                              ' all match after'
                              ' the first dimension (%s vs %s)') %
                              (inner_ds, ds[1:]))
    return coretypes.DataShape(*[coretypes.Fixed(outer_dim_size)] + list(inner_ds))


def collect(pred, expr):
    """ Collect terms in expression that match predicate

    >>> from datashader.datashape import Unit, dshape
    >>> predicate = lambda term: isinstance(term, Unit)
    >>> dshape = dshape('var * {value: int64, loc: 2 * int32}')
    >>> sorted(set(collect(predicate, dshape)), key=str)
    [Fixed(val=2), ctype("int32"), ctype("int64"), Var()]
    >>> from datashader.datashape import var, int64
    >>> sorted(set(collect(predicate, [var, int64])), key=str)
    [ctype("int64"), Var()]
    """
    if pred(expr):
        return [expr]
    if isinstance(expr, coretypes.Record):
        return chain.from_iterable(collect(pred, typ) for typ in expr.types)
    if isinstance(expr, coretypes.Mono):
        return chain.from_iterable(collect(pred, typ) for typ in expr.parameters)
    if isinstance(expr, (list, tuple)):
        return chain.from_iterable(collect(pred, item) for item in expr)


def has_var_dim(ds):
    """Returns True if datashape has a variable dimension

    Note currently treats variable length string as scalars.

    >>> has_var_dim(dshape('2 * int32'))
    False
    >>> has_var_dim(dshape('var * 2 * int32'))
    True
    """
    return has((coretypes.Ellipsis, coretypes.Var), ds)


def has(typ, ds):
    if isinstance(ds, typ):
        return True
    if isinstance(ds, coretypes.Record):
        return any(has(typ, t) for t in ds.types)
    if isinstance(ds, coretypes.Mono):
        return any(has(typ, p) for p in ds.parameters)
    if isinstance(ds, (list, tuple)):
        return any(has(typ, item) for item in ds)
    return False


def has_ellipsis(ds):
    """Returns True if the datashape has an ellipsis

    >>> has_ellipsis(dshape('2 * int'))
    False
    >>> has_ellipsis(dshape('... * int'))
    True
    """
    return has(coretypes.Ellipsis, ds)
