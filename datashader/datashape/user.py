from .dispatch import dispatch
from .coretypes import (
    CType, Date, DateTime, DataShape, Record, String, Time, Var, from_numpy, to_numpy_dtype)
from .predicates import isdimension
from .util import dshape
from datetime import date, time, datetime
import numpy as np


__all__ = ['validate', 'issubschema']


basetypes = np.generic, int, float, str, date, time, datetime


@dispatch(np.dtype, basetypes)
def validate(schema, value):
    return np.issubdtype(type(value), schema)


@dispatch(CType, basetypes)
def validate(schema, value):  # noqa: F811
    return validate(to_numpy_dtype(schema), value)


@dispatch(DataShape, (tuple, list))
def validate(schema, value):  # noqa: F811
    head = schema[0]
    return ((len(schema) == 1 and validate(head, value))
        or (isdimension(head)
       and (isinstance(head, Var) or int(head) == len(value))
       and all(validate(DataShape(*schema[1:]), item) for item in value)))


@dispatch(DataShape, object)
def validate(schema, value):  # noqa: F811
    if len(schema) == 1:
        return validate(schema[0], value)


@dispatch(Record, dict)
def validate(schema, d):  # noqa: F811
    return all(validate(sch, d.get(k)) for k, sch in schema.parameters[0])


@dispatch(Record, (tuple, list))
def validate(schema, seq):  # noqa: F811
    return all(validate(sch, item) for (k, sch), item
                                    in zip(schema.parameters[0], seq))


@dispatch(str, object)
def validate(schema, value):  # noqa: F811
    return validate(dshape(schema), value)


@dispatch(type, object)
def validate(schema, value):  # noqa: F811
    return isinstance(value, schema)


@dispatch(tuple, object)
def validate(schemas, value):  # noqa: F811
    return any(validate(schema, value) for schema in schemas)


@dispatch(object, object)
def validate(schema, value):  # noqa: F811
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
def issubschema(a, b):  # noqa: F811
    if a == b:
        return True
    # TODO, handle cases like float < real
    # TODO, handle records {x: int, y: int, z: int} < {x: int, y: int}

    return None  # We don't know, return something falsey
