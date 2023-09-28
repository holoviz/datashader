from __future__ import print_function, division, absolute_import
from datashape.dispatch import dispatch
from .coretypes import *
from .predicates import isdimension
from .util import dshape
import sys
from datetime import date, time, datetime


__all__ = ['validate', 'issubschema']


basetypes = np.generic, int, float, str, date, time, datetime


@dispatch(np.dtype, basetypes)
def validate(schema, value):
    return np.issubdtype(type(value), schema)


@dispatch(CType, basetypes)
def validate(schema, value):
    return validate(to_numpy_dtype(schema), value)


@dispatch(DataShape, (tuple, list))
def validate(schema, value):
    head = schema[0]
    return ((len(schema) == 1 and validate(head, value))
        or (isdimension(head)
       and (isinstance(head, Var) or int(head) == len(value))
       and all(validate(DataShape(*schema[1:]), item) for item in value)))


@dispatch(DataShape, object)
def validate(schema, value):
    if len(schema) == 1:
        return validate(schema[0], value)


@dispatch(Record, dict)
def validate(schema, d):
    return all(validate(sch, d.get(k)) for k, sch in schema.parameters[0])


@dispatch(Record, (tuple, list))
def validate(schema, seq):
    return all(validate(sch, item) for (k, sch), item
                                    in zip(schema.parameters[0], seq))


@dispatch(str, object)
def validate(schema, value):
    return validate(dshape(schema), value)


@dispatch(type, object)
def validate(schema, value):
    return isinstance(value, schema)


@dispatch(tuple, object)
def validate(schemas, value):
    return any(validate(schema, value) for schema in schemas)


@dispatch(object, object)
def validate(schema, value):
    return False


@validate.register(String, str)
@validate.register(Time, time)
@validate.register(Date, date)
@validate.register(DateTime, datetime)
def validate_always_true(schema, value):
    return True


@dispatch(DataShape, np.ndarray)
def validate(schema, value):
    return issubschema(from_numpy(value.shape, value.dtype), schema)


@dispatch(object, object)
def issubschema(a, b):
    return issubschema(dshape(a), dshape(b))


@dispatch(DataShape, DataShape)
def issubschema(a, b):
    if a == b:
        return True
    # TODO, handle cases like float < real
    # TODO, handle records {x: int, y: int, z: int} < {x: int, y: int}

    return None  # We don't know, return something falsey
