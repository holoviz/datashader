from __future__ import absolute_import, division, print_function

import re

from blaze.expr import Expr, summary, common_subexpression
from blaze.expr.split_apply_combine import _names_and_types

from datashape import dshape, DataShape, Record, Tuple, Option, Unit
from datashape.typesets import real
from datashape.predicates import launder


def isreal(dt):
    dt = launder(dt)
    return isinstance(dt, Unit) and dt in real


class Canvas(object):
    def __init__(self, plot_width=600, plot_height=600,
                 x_range=None, y_range=None, stretch=False):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.x_range = x_range
        self.y_range = y_range
        self.stretch = stretch

    def points(self, x, y, **kwargs):
        return bypixel(point(x, y), self, **kwargs)


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
        height = self.canvas.plot_height
        width = self.canvas.plot_width
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
