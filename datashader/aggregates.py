from __future__ import division, absolute_import, print_function

from datashape import dshape, Record, DataShape

from .core import Axis
from .utils import dshape_from_dynd


class Aggregate(object):
    def __repr__(self):
        return "{0}<dshape='{1}', shape={2}>".format(type(self).__name__,
                                                     self.dshape, self.shape)


def _validate_axis(axis):
    if not isinstance(axis, Axis):
        raise TypeError("axis must be instance of Axis")
    return axis


class ScalarAggregate(Aggregate):
    def __init__(self, data, x_axis=None, y_axis=None):
        self._data = data
        self.x_axis = _validate_axis(x_axis)
        self.y_axis = _validate_axis(y_axis)

    @property
    def dshape(self):
        if not hasattr(self, '_dshape'):
            self._dshape = dshape_from_dynd(self._data.dtype)
        return self._dshape

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            self._shape = self._data.shape
        return self._shape


class ByCategoriesAggregate(Aggregate):
    def __init__(self, data, cats, x_axis=None, y_axis=None):
        self._data = data
        self._cats = cats
        self.x_axis = _validate_axis(x_axis)
        self.y_axis = _validate_axis(y_axis)

    @property
    def dshape(self):
        if not hasattr(self, '_dshape'):
            self._dshape = DataShape(len(self._cats), 'int32')
        return self._dshape

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            self._shape = self._data.shape[:2]
        return self._shape


class RecordAggregate(Aggregate):
    def __init__(self, data, x_axis=None, y_axis=None):
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        if not data:
            raise ValueError("Empty data dictionary")
        aggs = list(data.values())
        shape = aggs[0].shape
        for a in aggs:
            if a.x_axis != x_axis or a.y_axis != y_axis or a.shape != shape:
                raise ValueError("Aggregates must have same shape and axes")
        self._data = data
        self.x_axis = _validate_axis(x_axis)
        self.y_axis = _validate_axis(y_axis)
        self._shape = shape

    @property
    def dshape(self):
        if not hasattr(self, '_dshape'):
            self._dshape = dshape(Record([(k, v.dshape) for (k, v) in
                                          self._data.items()]))
        return self._dshape

    @property
    def shape(self):
        return self._shape

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__) +
                      list(self.keys())))

    def __getattr__(self, key):
        if key in self._data:
            return self._data[key]
        else:
            raise AttributeError("'RecordAggregate' object has no attribute"
                                 "'{0}'".format(key))
