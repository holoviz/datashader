from __future__ import division, absolute_import, print_function

import operator

import numpy as np
from datashape import dshape, Record, DataShape, isboolean
from dynd import nd
from toolz import get

from .core import Axis
from .utils import (dshape_from_dynd, is_valid_identifier, is_option,
                    dynd_missing_types, dynd_to_np_mask)


__all__ = ['ScalarAggregate', 'CategoricalAggregate', 'RecordAggregate']


class Aggregate(object):
    def __repr__(self):
        return "{0}<dshape='{1}', shape={2}>".format(type(self).__name__,
                                                     self.dshape, self.shape)


def _validate_axis(axis):
    if not isinstance(axis, Axis):
        raise TypeError("axis must be instance of Axis")
    return axis


def _validate_aligned(a, b):
    if (a.x_axis != b.x_axis or a.y_axis != b.y_axis or a.shape != b.shape):
        raise NotImplementedError("operations between aggregates with "
                                  "non-matching axis or shape")


def dynd_op(op, left, right):
    if isinstance(left, nd.array):
        left_np, left_missing = dynd_to_np_mask(left)
        left_option = is_option(left.dtype)
    else:
        left_np, left_missing = left, False
        left_option = False
    if isinstance(right, nd.array):
        right_np, right_missing = dynd_to_np_mask(right)
        right_option = is_option(right.dtype)
    else:
        right_np, right_missing = right, False
        right_option = False
    out = op(left_np, right_np)
    if left_option or right_option:
        if out.dtype in dynd_missing_types:
            out[left_missing | right_missing] = dynd_missing_types[out.dtype]
            out = nd.asarray(out)
            return nd.asarray(out).view_scalars('?' + str(out.dtype))
        else:
            raise ValueError("Missing type unknown")
    return nd.asarray(out)


def make_binary_op(op, use_dynd=True):
    agg_op = op if use_dynd else lambda l, r: dynd_op(op, l, r)
    def f(self, other):
        if isinstance(other, ScalarAggregate):
            _validate_aligned(self, other)
            res = agg_op(self._data, other._data)
        elif isinstance(other, (np.generic, int, float, bool)):
            res = agg_op(self._data, other)
        else:
            return NotImplemented
        return ScalarAggregate(res, self.x_axis, self.y_axis)
    return f


def make_unary_op(op):
    def f(self):
        arr, missing = dynd_to_np_mask(self._data)
        out = op(arr)
        if is_option(self._data.dtype):
            out[missing] = dynd_missing_types[out.dtype]
            out = nd.asarray(out).view_scalars('?' + str(out.dtype))
        else:
            out = nd.asarray(out)
        return ScalarAggregate(out, self.x_axis, self.y_axis)
    return f


def right(method):
    """Wrapper to create 'right' version of operator given left version"""
    def _inner(self, other):
        return method(other, self)
    return _inner


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

    def where(self, cond, otherwise=None):
        """Return elements from `self` or `otherwise`, depending on `cond`.

        Parameters
        ----------
        cond : ScalarAggregate
            A boolean `ScalarAggregate` of the same shape as `self`.
        otherwise : ScalarAggregate, numeric, None, optional
            The value to use in each element if `cond` is `False`. If an
            aggregate, must have same shape as `self` and `cond`. If `None`
            [default], yields a missing value for the given datashape.

        Returns
        -------
        ScalarAggregate
        """
        if not isinstance(cond, ScalarAggregate):
            raise TypeError("cond must be instance of ScalarAggregate")
        out = _where_helper(self, cond, otherwise)
        return ScalarAggregate(out, self.x_axis, self.y_axis)

    def interpolate(self, low, high, **kwargs):
        """Convert a ScalarAggregate to an image.

        Parameters
        ----------
        low : color name or tuple
            The color for the low end of the scale. Can be specified either by
            name, hexcode, or as a tuple of ``(red, green, blue)`` values.
        high : color name or tuple
            The color for the high end of the scale
        how : string or callable
            The interpolation method to use. Valid strings are 'log' [default],
            'cbrt', and 'linear'. Callables take a 2-dimensional array of
            magnitudes at each pixel, and should return a numeric array of the
            same shape.
        """
        from .transfer_functions import interpolate
        return interpolate(self, low, high, **kwargs)

    __add__ = make_binary_op(operator.add)
    __sub__ = make_binary_op(operator.sub)
    __mul__ = make_binary_op(operator.mul)
    __floordiv__ = make_binary_op(operator.floordiv, False)
    __div__ = __floordiv__
    __truediv__ = make_binary_op(operator.truediv, False)
    __eq__ = make_binary_op(operator.eq)
    __ne__ = make_binary_op(operator.ne)
    __lt__ = make_binary_op(operator.lt)
    __le__ = make_binary_op(operator.le)
    __gt__ = make_binary_op(operator.gt)
    __ge__ = make_binary_op(operator.ge)
    __and__ = make_binary_op(operator.and_, False)
    __or__ = make_binary_op(operator.or_, False)
    __xor__ = make_binary_op(operator.xor, False)
    __radd__ = make_binary_op(right(operator.add))
    __rsub__ = make_binary_op(right(operator.sub))
    __rmul__ = make_binary_op(right(operator.mul))
    __rfloordiv__ = make_binary_op(right(operator.floordiv), False)
    __rdiv__ = __rfloordiv__
    __rtruediv__ = make_binary_op(right(operator.truediv), False)
    __req__ = make_binary_op(right(operator.eq))
    __rne__ = make_binary_op(right(operator.ne))
    __rlt__ = make_binary_op(right(operator.lt))
    __rle__ = make_binary_op(right(operator.le))
    __rgt__ = make_binary_op(right(operator.gt))
    __rge__ = make_binary_op(right(operator.ge))
    __rand__ = make_binary_op(right(operator.and_), False)
    __ror__ = make_binary_op(right(operator.or_), False)
    __rxor__ = make_binary_op(right(operator.xor), False)
    __abs__ = make_unary_op(operator.abs)
    __neg__ = make_unary_op(operator.neg)
    __pos__ = make_unary_op(operator.pos)
    __invert__ = make_unary_op(operator.inv)


class CategoricalAggregate(Aggregate):
    def __init__(self, data, cats, x_axis=None, y_axis=None):
        self._data = data
        self._cats = cats
        self.x_axis = _validate_axis(x_axis)
        self.y_axis = _validate_axis(y_axis)

    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__) +
                      list(filter(is_valid_identifier, self._cats))))

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("'CategoricalAggregate' object has no "
                                 "attribute '{0}'".format(key))

    def __getitem__(self, key):
        try:
            if isinstance(key, list):
                # List of categories
                inds = [self._cats.index(k) for k in key]
                dtype = self._data.dtype
                if is_option(dtype):
                    out = nd.as_numpy(self._data.view_scalars(
                                      dtype.value_type))
                else:
                    out = nd.as_numpy(self._data)
                out = nd.asarray(out[:, :, inds]).view_scalars(dtype)
                return CategoricalAggregate(out, key, self.x_axis, self.y_axis)
            else:
                # Single category
                i = self._cats.index(key)
                return ScalarAggregate(self._data[:, :, i],
                                       self.x_axis, self.y_axis)
        except ValueError:
            raise KeyError("'{0}'".format(key))

    @property
    def cats(self):
        return self._cats

    @property
    def dshape(self):
        if not hasattr(self, '_dshape'):
            self._dshape = DataShape(len(self._cats),
                                     dshape_from_dynd(self._data.dtype))
        return self._dshape

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            self._shape = self._data.shape[:2]
        return self._shape

    def where(self, cond, otherwise=None):
        """Return elements from `self` or `otherwise`, depending on `cond`.

        Parameters
        ----------
        cond : CategoricalAggregate, ScalarAggregate
            A boolean aggregate of the same shape as `self`.
        otherwise : Aggregate, numeric, None, optional
            The value to use in each element if `cond` is `False`. If an
            aggregate, must have same shape as `self` and `cond`. If `None`
            [default], yields a missing value for the given datashape.

        Returns
        -------
        CategoricalAggregate
        """
        if (self.cats != getattr(cond, 'cats', self.cats) !=
                getattr(otherwise, 'cats', self.cats)):
            raise ValueError("Categoricals not aligned")
        if not isinstance(cond, (CategoricalAggregate, ScalarAggregate)):
            raise TypeError("cond must be instance of ScalarAggregate "
                            "or CategoricalAggregate")
        out = _where_helper(self, cond, otherwise)
        return CategoricalAggregate(out, self.cats, self.x_axis, self.y_axis)

    def colorize(self, color_key, **kwargs):
        """Color a CategoricalAggregate by field.

        Parameters
        ----------
        color_key : dict or iterable
            A mapping of fields to colors. Can be either a ``dict`` mapping
            from field name to colors, or an iterable of colors in the same
            order as the record fields.
        how : string or callable
            The interpolation method to use. Valid strings are 'log' [default],
            'cbrt', and 'linear'. Callables take a 2-dimensional array of
            magnitudes at each pixel, and should return a numeric array of the
            same shape.
        min_alpha : float, optional
            The minimum alpha value to use for non-empty pixels, in [0, 255].
        """
        from .transfer_functions import colorize
        return colorize(self, color_key, **kwargs)


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
                                          sorted(self._data.items())]))
        return self._dshape

    @property
    def shape(self):
        return self._shape

    def keys(self):
        return list(sorted(self._data.keys()))

    def values(self):
        return list(sorted(self._data.values()))

    def items(self):
        return list(sorted(self._data.items()))

    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__) +
                      list(self.keys())))

    def __getitem__(self, key):
        if isinstance(key, list):
            return RecordAggregate(dict(zip(key, get(key, self._data))),
                                   self.x_axis, self.y_axis)
        return self._data[key]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("'RecordAggregate' object has no attribute"
                                 "'{0}'".format(key))


def _where_helper(agg, cond, otherwise):
    if not isboolean(cond.dshape):
        raise TypeError("cond must be a boolean aggregate")
    _validate_aligned(agg, cond)
    cond_arr, _ = dynd_to_np_mask(cond._data)
    arr, arr_mask = dynd_to_np_mask(agg._data)
    arr_missing = is_option(agg._data.dtype)
    while cond_arr.ndim < arr.ndim:
        cond_arr = np.expand_dims(cond_arr, -1)
    if isinstance(otherwise, Aggregate):
        _validate_aligned(agg, otherwise)
        otherwise_arr, otherwise_mask = dynd_to_np_mask(otherwise._data)
        while otherwise_arr.ndim < arr.ndim:
            otherwise_arr = np.expand_dims(otherwise_arr, -1)
            otherwise_mask = np.expand_dims(otherwise_mask, -1)
        otherwise_missing = is_option(otherwise._data.dtype)
    elif isinstance(otherwise, (int, float, np.generic)):
        otherwise_arr = otherwise
        otherwise_mask = otherwise_missing = False
    elif otherwise is None:
        otherwise_arr = dynd_missing_types[arr.dtype]
        otherwise_mask = False
        otherwise_missing = True
    else:
        raise TypeError("`otherwise` got unknown"
                        " type: {0}".format(type(otherwise)))
    out = np.where(cond_arr, arr, otherwise_arr)
    if arr_missing or otherwise_missing:
        out[arr_mask | otherwise_mask] = dynd_missing_types[out.dtype]
        out = nd.asarray(out)
        out = out.view_scalars('?' + str(out.dtype))
    else:
        out = nd.asarray(out)
    return out
