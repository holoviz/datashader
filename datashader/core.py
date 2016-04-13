from __future__ import absolute_import, division, print_function

import numpy as np
from datashape.predicates import istabular
from odo import discover

from .utils import Dispatcher, ngjit


class Expr(object):
    """Base class for expression-like objects.

    Implements hashing and equality checks. Subclasses should implement an
    ``inputs`` attribute/property, containing a tuple of everything that fully
    defines that expression.
    """
    def __hash__(self):
        return hash((type(self), self.inputs))

    def __eq__(self, other):
        return type(self) is type(other) and self.inputs == other.inputs

    def __ne__(self, other):
        return not self == other


class Axis(object):
    """Interface for implementing axis transformations.

    Instances hold implementations of transformations to and from axis space.
    The default implementation is equivalent to:

    >>> def forward_transform(data_x):
    ...     scale * mapper(data_x) + t
    >>> def inverse_transform(axis_x):
    ...     inverse_mapper((axis_x - t)/s)

    Where ``mapper`` and ``inverse_mapper`` are elementwise functions mapping
    to and from axis-space respectively, and ``scale`` and ``transform`` are
    parameters describing a linear scale and translate transformation, computed
    by the ``compute_scale_and_translate`` method.

    Parameters
    ----------
    mapper : callable
        A mapping from data space to axis space
    inverse_mapper : callable
        A mapping from axis space to data space
    """
    def __init__(self, mapper, inverse_mapper):
        self.mapper = mapper
        self.inverse_mapper = inverse_mapper

    def compute_scale_and_translate(self, range, n):
        """Compute the scale and translate parameters for a linear transformation
        ``output = s * input + t``, mapping from data space to axis space.

        Parameters
        ----------
        range : tuple
            A tuple representing the boundary inclusive range ``[min, max]``
            along the axis, in data space.
        n : int
            The number of bins along the axis.

        Returns
        -------
        s, t : floats
            Parameters represe

        """
        start, end = map(self.mapper, range)
        s = (n - 1)/(end - start)
        t = -start * s
        return s, t

    def compute_index(self, st, n):
        """Compute a 1D array representing the axis index.

        Parameters
        ----------
        st : tuple
            A tuple of ``(scale, translate)`` parameters.
        n : int
            The number of bins along the dimension.

        Returns
        -------
        index : ndarray
        """
        px = np.arange(n)
        s, t = st
        return self.inverse_mapper((px - t)/s)


_axis_lookup = {'linear': Axis(ngjit(lambda x: x), ngjit(lambda x: x)),
                'log': Axis(ngjit(lambda x: np.log10(x)),
                            ngjit(lambda x: 10**x))}


class Canvas(object):
    """An abstract canvas representing the space in which to bin.

    Parameters
    ----------
    plot_width, plot_height : int, optional
        Width and height of the output aggregate in pixels.
    x_range, y_range : tuple, optional
        A tuple representing the bounds inclusive space ``[min, max]`` along
        the axis.
    x_axis_type, y_axis_type : str, optional
        The type of the axis. Valid options are ``'linear'`` [default], and
        ``'log'``.
    """
    def __init__(self, plot_width=600, plot_height=600,
                 x_range=None, y_range=None,
                 x_axis_type='linear', y_axis_type='linear'):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.x_range = tuple(x_range) if x_range else x_range
        self.y_range = tuple(y_range) if y_range else y_range
        self.x_axis = _axis_lookup[x_axis_type]
        self.y_axis = _axis_lookup[y_axis_type]

    def points(self, source, x, y, agg=None):
        """Compute a reduction by pixel, mapping data to pixels as points.

        Parameters
        ----------
        source : pandas.DataFrame, dask.DataFrame
            The input datasource.
        x, y : str
            Column names for the x and y coordinates of each point.
        agg : Reduction, optional
            Reduction to compute. Default is ``count()``.
        """
        from .glyphs import Point
        from .reductions import count
        if agg is None:
            agg = count()
        return bypixel(source, self, Point(x, y), agg)

    def line(self, source, x, y, agg=None):
        """Compute a reduction by pixel, mapping data to pixels as a line.

        For aggregates that take in extra fields, the interpolated bins will
        receive the fields from the previous point. In pseudocode:

        >>> for i in range(len(rows) - 1):    # doctest: +SKIP
        ...     row0 = rows[i]
        ...     row1 = rows[i + 1]
        ...     for xi, yi in interpolate(row0.x, row0.y, row1.x, row1.y):
        ...         add_to_aggregate(xi, yi, row0)

        Parameters
        ----------
        source : pandas.DataFrame, dask.DataFrame
            The input datasource.
        x, y : str
            Column names for the x and y coordinates of each vertex.
        agg : Reduction, optional
            Reduction to compute. Default is ``any()``.
        """
        from .glyphs import Line
        from .reductions import any
        if agg is None:
            agg = any()
        return bypixel(source, self, Line(x, y), agg)


def bypixel(source, canvas, glyph, agg):
    """Compute an aggregate grouped by pixel sized bins.

    Aggregate input data ``source`` into a grid with shape and axis matching
    ``canvas``, mapping data to bins by ``glyph``, and aggregating by reduction
    ``agg``.

    Parameters
    ----------
    source : pandas.DataFrame, dask.DataFrame
        Input datasource
    canvas : Canvas
    glyph : Glyph
    agg : Reduction
    """
    dshape = discover(source)
    if not istabular(dshape):
        raise ValueError("source must be tabular")
    schema = dshape.measure
    glyph.validate(schema)
    agg.validate(schema)
    return bypixel.pipeline(source, schema, canvas, glyph, agg)


bypixel.pipeline = Dispatcher()
