from __future__ import absolute_import, division, print_function

import re

from blaze import symbol
from blaze.expr import Expr, summary, common_subexpression
from blaze.expr.split_apply_combine import _names_and_types

from datashape import dshape, DataShape, Record, Tuple, Option, var, isreal


class Canvas(Expr):
    """Record representing a canvas in space"""
    __slots__ = '_hash', 'width', 'height', 'x_range', 'y_range'
    __inputs__ = 'width', 'height', 'x_range', 'y_range'
    schema = dshape('{width: int64, height: int64, '
                    'x_range: (float64, float64), '
                    'y_range: (float64, float64)}')

    def _dshape(self):
        return self.schema

    def __repr__(self):
        return str(self)


def canvas(width=None, height=None, x_range=None, y_range=None):
    if width is None:
        width = symbol('width', 'int64')
    if height is None:
        height = symbol('height', 'int64')
    if not x_range:
        x_range = symbol('x_range', '(float64, float64)')
    if not y_range:
        y_range = symbol('y_range', '(float64, float64)')
    return Canvas(width, height, x_range, y_range)


class Glyph(Expr):
    """Record representing a shape in space"""
    @property
    def _child(self):
        return common_subexpression(*self._args)

    def _dshape(self):
        return self.schema


class Point(Glyph):
    __slots__ = '_hash', 'x', 'y'
    schema = dshape('{x: float64, y: float64}')


def point(x, y):
    if not isreal(x.schema):
        raise TypeError('x.schema must be real')
    elif not isreal(y.schema):
        raise TypeError('y.schema must be real')
    return Point(x, y)


class Circle(Glyph):
    __slots__ = '_hash', 'x', 'y', 'r'
    schema = dshape('{x: float64, y: float64, r: float64}')


def circle(x, y, r):
    if not isreal(x.schema):
        raise TypeError('x.schema must be real')
    elif not isreal(y.schema):
        raise TypeError('y.schema must be real')
    elif not isreal(r.schema):
        raise TypeError('r.schema must be real')
    return Circle(x, y, r)


def optionify(d):
    """Make all datashape leaf nodes optional"""
    if isinstance(d, DataShape):
        return DataShape(*(optionify(i) for i in d.parameters))
    elif isinstance(d, Record):
        return Record(tuple((f, optionify(v)) for (f, v) in
                      zip(d.names, d.types)))
    elif isinstance(d, Tuple):
        return Tuple(tuple(optionify(i) for i in d.dshapes))
    elif isinstance(d, Option):
        return d
    return Option(d)


class ByPixel(Expr):
    __slots__ = '_hash', 'glyph', 'canvas', 'apply'

    @property
    def _child(self):
        return common_subexpression(self.glyph, self.apply)

    def _schema(self):
        names, types = _names_and_types(self.apply)
        return dshape(Record(list(zip(names, map(optionify, types)))))

    def _dshape(self):
        if isinstance(self.canvas.width, Expr):
            width = var
        else:
            width = self.canvas.width
        if isinstance(self.canvas.height, Expr):
            height = var
        else:
            height = self.canvas.height
        return height * (width * self.schema)

    def __str__(self):
        return 'bypixel(%s, %s, %s)' % (self.glyph, self.canvas,
                                        re.sub(r'^summary\((.*)\)$', r'\1',
                                               str(self.apply)))


def bypixel(glyph, canvas, **kwargs):
    if not isinstance(glyph, Glyph):
        raise TypeError('glyph must be a Glyph')
    elif not isinstance(canvas, Canvas):
        raise TypeError('canvas must be a Canvas')
    return ByPixel(glyph, canvas, summary(**kwargs))
