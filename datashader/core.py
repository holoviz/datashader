from __future__ import absolute_import, division, print_function

from numbers import Number
from math import log10

import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from six import string_types
from xarray import DataArray, Dataset
from collections import OrderedDict

from .utils import Dispatcher, ngjit, calc_res, calc_bbox, orient_array, \
    compute_coords, dshape_from_xarray_dataset
from .utils import get_indices, dshape_from_pandas, dshape_from_dask
from .utils import Expr # noqa (API import)
from .resampling import resample_2d, resample_2d_distributed
from . import reductions as rd

try:
    import cudf
except Exception:
    cudf = None

try:
    import dask_cudf
except Exception:
    dask_cudf = None

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
    """

    def compute_scale_and_translate(self, range, n):
        """Compute the scale and translate parameters for a linear transformation
        ``output = s * input + t``, mapping from data space to axis space.

        Parameters
        ----------
        range : tuple
            A tuple representing the range ``[min, max]`` along the axis, in
            data space. Both min and max are inclusive.
        n : int
            The number of bins along the axis.

        Returns
        -------
        s, t : floats
        """
        start, end = map(self.mapper, range)
        s = n/(end - start)
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
        px = np.arange(n)+0.5
        s, t = st
        return self.inverse_mapper((px - t)/s)

    def mapper(val):
        """A mapping from data space to axis space"""
        raise NotImplementedError

    def inverse_mapper(val):
        """A mapping from axis space to data space"""
        raise NotImplementedError

    def validate(self, range):
        """Given a range (low,high), raise an error if the range is invalid for this axis"""
        pass


class LinearAxis(Axis):
    """A linear Axis"""
    @staticmethod
    @ngjit
    def mapper(val):
        return val

    @staticmethod
    @ngjit
    def inverse_mapper(val):
        return val


class LogAxis(Axis):
    """A base-10 logarithmic Axis"""
    @staticmethod
    @ngjit
    def mapper(val):
        return log10(float(val))

    @staticmethod
    @ngjit
    def inverse_mapper(val):
        y = 10 # temporary workaround for https://github.com/numba/numba/issues/3135 (numba 0.39.0)
        return y**val

    def validate(self, range):
        low, high = map(self.mapper, range)
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError('Range values must be >0 for a LogAxis')


_axis_lookup = {'linear': LinearAxis(), 'log': LogAxis()}


def validate_xy_or_geometry(glyph, x, y, geometry):
    if (geometry is None and (x is None or y is None) or
            geometry is not None and (x is not None or y is not None)):
        raise ValueError("""
{glyph} coordinates may be specified by providing both the x and y arguments, or by
providing the geometry argument. Received:
    x: {x}
    y: {y}
    geometry: {geometry}
""".format(glyph=glyph, x=repr(x), y=repr(y), geometry=repr(geometry)))


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
        self.x_range = None if x_range is None else tuple(x_range)
        self.y_range = None if y_range is None else tuple(y_range)
        self.x_axis = _axis_lookup[x_axis_type]
        self.y_axis = _axis_lookup[y_axis_type]

    def points(self, source, x=None, y=None, agg=None, geometry=None):
        """Compute a reduction by pixel, mapping data to pixels as points.

        Parameters
        ----------
        source : pandas.DataFrame, dask.DataFrame, or xarray.DataArray/Dataset
            The input datasource.
        x, y : str
            Column names for the x and y coordinates of each point. If provided,
            the geometry argument may not also be provided.
        agg : Reduction, optional
            Reduction to compute. Default is ``count()``.
        geometry: str
            Column name of a PointsArray of the coordinates of each point. If provided,
            the x and y arguments may not also be provided.
        """
        from .glyphs import Point, MultiPointGeometry
        from .reductions import count as count_rdn

        validate_xy_or_geometry('Point', x, y, geometry)

        if agg is None:
            agg = count_rdn()

        if geometry is None:
            glyph = Point(x, y)
        else:
            from spatialpandas import GeoDataFrame
            from spatialpandas.dask import DaskGeoDataFrame
            if isinstance(source, DaskGeoDataFrame):
                # Downselect partitions to those that may contain points in viewport
                x_range = self.x_range if self.x_range is not None else (None, None)
                y_range = self.y_range if self.y_range is not None else (None, None)
                source = source.cx_partitions[slice(*x_range), slice(*y_range)]
            elif not isinstance(source, GeoDataFrame):
                raise ValueError(
                    "source must be an instance of spatialpandas.GeoDataFrame or \n"
                    "spatialpandas.dask.DaskGeoDataFrame.\n"
                    "  Received value of type {typ}".format(typ=type(source)))

            glyph = MultiPointGeometry(geometry)

        return bypixel(source, self, glyph, agg)

    def line(self, source, x=None, y=None, agg=None, axis=0, geometry=None):
        """Compute a reduction by pixel, mapping data to pixels as one or
        more lines.

        For aggregates that take in extra fields, the interpolated bins will
        receive the fields from the previous point. In pseudocode:

        >>> for i in range(len(rows) - 1):    # doctest: +SKIP
        ...     row0 = rows[i]
        ...     row1 = rows[i + 1]
        ...     for xi, yi in interpolate(row0.x, row0.y, row1.x, row1.y):
        ...         add_to_aggregate(xi, yi, row0)

        Parameters
        ----------
        source : pandas.DataFrame, dask.DataFrame, or xarray.DataArray/Dataset
            The input datasource.
        x, y : str or number or list or tuple or np.ndarray
            Specification of the x and y coordinates of each vertex
            * str or number: Column labels in source
            * list or tuple: List or tuple of column labels in source
            * np.ndarray: When axis=1, a literal array of the
              coordinates to be used for every row
        agg : Reduction, optional
            Reduction to compute. Default is ``any()``.
        axis : 0 or 1, default 0
            Axis in source to draw lines along
            * 0: Draw lines using data from the specified columns across
                 all rows in source
            * 1: Draw one line per row in source using data from the
                 specified columns
        geometry : str
            Column name of a LinesArray of the coordinates of each line. If provided,
            the x and y arguments may not also be provided.

        Examples
        --------
        Define a canvas and a pandas DataFrame with 6 rows
        >>> import pandas as pd  # doctest: +SKIP
        ... import numpy as np
        ... import datashader as ds
        ... from datashader import Canvas
        ... import datashader.transfer_functions as tf
        ... cvs = Canvas()
        ... df = pd.DataFrame({
        ...    'A1': [1, 1.5, 2, 2.5, 3, 4],
        ...    'A2': [1.5, 2, 3, 3.2, 4, 5],
        ...    'B1': [10, 12, 11, 14, 13, 15],
        ...    'B2': [11, 9, 10, 7, 8, 12],
        ... }, dtype='float64')

        Aggregate one line across all rows, with coordinates df.A1 by df.B1
        >>> agg = cvs.line(df, x='A1', y='B1', axis=0)  # doctest: +SKIP
        ... tf.spread(tf.shade(agg))

        Aggregate two lines across all rows. The first with coordinates
        df.A1 by df.B1 and the second with coordinates df.A2 by df.B2
        >>> agg = cvs.line(df, x=['A1', 'A2'], y=['B1', 'B2'], axis=0)  # doctest: +SKIP
        ... tf.spread(tf.shade(agg))

        Aggregate two lines across all rows where the lines share the same
        x coordinates. The first line will have coordinates df.A1 by df.B1
        and the second will have coordinates df.A1 by df.B2
        >>> agg = cvs.line(df, x='A1', y=['B1', 'B2'], axis=0)  # doctest: +SKIP
        ... tf.spread(tf.shade(agg))

        Aggregate 6 length-2 lines, one per row, where the ith line has
        coordinates [df.A1[i], df.A2[i]] by [df.B1[i], df.B2[i]]
        >>> agg = cvs.line(df, x=['A1', 'A2'], y=['B1', 'B2'], axis=1)  # doctest: +SKIP
        ... tf.spread(tf.shade(agg))

        Aggregate 6 length-4 lines, one per row, where the x coordinates
        of every line are [0, 1, 2, 3] and the y coordinates of the ith line
        are [df.A1[i], df.A2[i], df.B1[i], df.B2[i]].
        >>> agg = cvs.line(df,  # doctest: +SKIP
        ...                x=np.arange(4),
        ...                y=['A1', 'A2', 'B1', 'B2'],
        ...                axis=1)
        ... tf.spread(tf.shade(agg))

        Aggregate RaggedArrays of variable length lines, one per row
        (requires pandas >= 0.24.0)
        >>> df_ragged = pd.DataFrame({  # doctest: +SKIP
        ...    'A1': pd.array([
        ...        [1, 1.5], [2, 2.5, 3], [1.5, 2, 3, 4], [3.2, 4, 5]],
        ...        dtype='Ragged[float32]'),
        ...    'B1': pd.array([
        ...        [10, 12], [11, 14, 13], [10, 7, 9, 10], [7, 8, 12]],
        ...        dtype='Ragged[float32]'),
        ...    'group': pd.Categorical([0, 1, 2, 1])
        ... })
        ...
        ... agg = cvs.line(df_ragged, x='A1', y='B1', axis=1)
        ... tf.spread(tf.shade(agg))

        Aggregate RaggedArrays of variable length lines by group column,
        one per row (requires pandas >= 0.24.0)
        >>> agg = cvs.line(df_ragged, x='A1', y='B1',  # doctest: +SKIP
        ...                agg=ds.count_cat('group'), axis=1)
        ... tf.spread(tf.shade(agg))
        """
        from .glyphs import (LineAxis0, LinesAxis1, LinesAxis1XConstant,
                             LinesAxis1YConstant, LineAxis0Multi,
                             LinesAxis1Ragged, LineAxis1Geometry)
        from .reductions import any as any_rdn

        validate_xy_or_geometry('Line', x, y, geometry)

        if agg is None:
            agg = any_rdn()

        if geometry is not None:
            from spatialpandas import GeoDataFrame
            from spatialpandas.dask import DaskGeoDataFrame
            if isinstance(source, DaskGeoDataFrame):
                # Downselect partitions to those that may contain lines in viewport
                x_range = self.x_range if self.x_range is not None else (None, None)
                y_range = self.y_range if self.y_range is not None else (None, None)
                source = source.cx_partitions[slice(*x_range), slice(*y_range)]
            elif not isinstance(source, GeoDataFrame):
                raise ValueError(
                    "source must be an instance of spatialpandas.GeoDataFrame or \n"
                    "spatialpandas.dask.DaskGeoDataFrame.\n"
                    "  Received value of type {typ}".format(typ=type(source)))

            glyph = LineAxis1Geometry(geometry)
        else:
            # Broadcast column specifications to handle cases where
            # x is a list and y is a string or vice versa
            orig_x, orig_y = x, y
            x, y = _broadcast_column_specifications(x, y)

            if axis == 0:
                if (isinstance(x, (Number, string_types)) and
                        isinstance(y, (Number, string_types))):
                    glyph = LineAxis0(x, y)
                elif (isinstance(x, (list, tuple)) and
                        isinstance(y, (list, tuple))):
                    glyph = LineAxis0Multi(tuple(x), tuple(y))
                else:
                    raise ValueError("""
Invalid combination of x and y arguments to Canvas.line when axis=0.
    Received:
        x: {x}
        y: {y}
See docstring for more information on valid usage""".format(
                        x=repr(orig_x), y=repr(orig_y)))

            elif axis == 1:
                if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
                    glyph = LinesAxis1(tuple(x), tuple(y))
                elif (isinstance(x, np.ndarray) and
                      isinstance(y,  (list, tuple))):
                    glyph = LinesAxis1XConstant(x, tuple(y))
                elif (isinstance(x, (list, tuple)) and
                      isinstance(y, np.ndarray)):
                    glyph = LinesAxis1YConstant(tuple(x), y)
                elif (isinstance(x, (Number, string_types)) and
                        isinstance(y, (Number, string_types))):
                    glyph = LinesAxis1Ragged(x, y)
                else:
                    raise ValueError("""
Invalid combination of x and y arguments to Canvas.line when axis=1.
    Received:
        x: {x}
        y: {y}
See docstring for more information on valid usage""".format(
                        x=repr(orig_x), y=repr(orig_y)))

            else:
                raise ValueError("""
The axis argument to Canvas.line must be 0 or 1
    Received: {axis}""".format(axis=axis))

        return bypixel(source, self, glyph, agg)

    def area(self, source, x, y, agg=None, axis=0, y_stack=None):
        """Compute a reduction by pixel, mapping data to pixels as a filled
        area region

        Parameters
        ----------
        source : pandas.DataFrame, dask.DataFrame, or xarray.DataArray/Dataset
            The input datasource.
        x, y : str or number or list or tuple or np.ndarray
            Specification of the x and y coordinates of each vertex of the
            line defining the starting edge of the area region.
            * str or number: Column labels in source
            * list or tuple: List or tuple of column labels in source
            * np.ndarray: When axis=1, a literal array of the
              coordinates to be used for every row
        agg : Reduction, optional
            Reduction to compute. Default is ``count()``.
        axis : 0 or 1, default 0
            Axis in source to draw lines along
            * 0: Draw area regions using data from the specified columns
                 across all rows in source
            * 1: Draw one area region per row in source using data from the
                 specified columns
        y_stack: str or number or list or tuple or np.ndarray or None
            Specification of the y coordinates of each vertex of the line
            defining the ending edge of the area region, where the x
            coordinate is given by the x argument described above.

            If y_stack is None, then the area region is filled to the y=0 line

            If y_stack is not None, then the form of y_stack must match the
            form of y.

        Examples
        --------
        Define a canvas and a pandas DataFrame with 6 rows
        >>> import pandas as pd  # doctest: +SKIP
        ... import numpy as np
        ... import datashader as ds
        ... from datashader import Canvas
        ... import datashader.transfer_functions as tf
        ... cvs = Canvas()
        ... df = pd.DataFrame({
        ...    'A1': [1, 1.5, 2, 2.5, 3, 4],
        ...    'A2': [1.6, 2.1, 2.9, 3.2, 4.2, 5],
        ...    'B1': [10, 12, 11, 14, 13, 15],
        ...    'B2': [11, 9, 10, 7, 8, 12],
        ... }, dtype='float64')

        Aggregate one area region across all rows, that starts with
        coordinates df.A1 by df.B1 and is filled to the y=0 line
        >>> agg = cvs.area(df, x='A1', y='B1',  # doctest: +SKIP
        ...                agg=ds.count(), axis=0)
        ... tf.shade(agg)

        Aggregate one area region across all rows, that starts with
        coordinates df.A1 by df.B1 and is filled to the line with coordinates
        df.A1 by df.B2
        >>> agg = cvs.area(df, x='A1', y='B1', y_stack='B2', # doctest: +SKIP
        ...                agg=ds.count(), axis=0)
        ... tf.shade(agg)

        Aggregate two area regions across all rows. The first starting
        with coordinates df.A1 by df.B1 and the second with coordinates
        df.A2 by df.B2. Both regions are filled to the y=0 line
        >>> agg = cvs.area(df, x=['A1', 'A2'], y=['B1', 'B2'], agg=ds.count(), axis=0)  # doctest: +SKIP
        ... tf.shade(agg)

        Aggregate two area regions across all rows where the regions share the
        same x coordinates. The first region will start with coordinates
        df.A1 by df.B1 and the second will start with coordinates
        df.A1 by df.B2. Both regions are filled to the y=0 line
        >>> agg = cvs.area(df, x='A1', y=['B1', 'B2'], agg=ds.count(), axis=0)  # doctest: +SKIP
        ... tf.shade(agg)

        Aggregate 6 length-2 area regions, one per row, where the ith region
        starts with coordinates [df.A1[i], df.A2[i]] by [df.B1[i], df.B2[i]]
        and is filled to the y=0 line
        >>> agg = cvs.area(df, x=['A1', 'A2'], y=['B1', 'B2'], agg=ds.count(), axis=1)  # doctest: +SKIP
        ... tf.shade(agg)

        Aggregate 6 length-4 area regions, one per row, where the
        starting x coordinates of every region are [0, 1, 2, 3] and
        the starting y coordinates of the ith region are
        [df.A1[i], df.A2[i], df.B1[i], df.B2[i]].  All regions are filled to
        the y=0 line
        >>> agg = cvs.area(df,  # doctest: +SKIP
        ...                x=np.arange(4),
        ...                y=['A1', 'A2', 'B1', 'B2'],
        ...                agg=ds.count(),
        ...                axis=1)
        ... tf.shade(agg)

        Aggregate RaggedArrays of variable length area regions, one per row.
        The starting coordinates of the ith region are df_ragged.A1 by
        df_ragged.B1 and the regions are filled to the y=0 line.
        (requires pandas >= 0.24.0)
        >>> df_ragged = pd.DataFrame({  # doctest: +SKIP
        ...    'A1': pd.array([
        ...        [1, 1.5], [2, 2.5, 3], [1.5, 2, 3, 4], [3.2, 4, 5]],
        ...        dtype='Ragged[float32]'),
        ...    'B1': pd.array([
        ...        [10, 12], [11, 14, 13], [10, 7, 9, 10], [7, 8, 12]],
        ...        dtype='Ragged[float32]'),
        ...    'B2': pd.array([
        ...        [6, 10], [9, 10, 18], [9, 5, 6, 8], [4, 5, 11]],
        ...        dtype='Ragged[float32]'),
        ...    'group': pd.Categorical([0, 1, 2, 1])
        ... })
        ...
        ... agg = cvs.area(df_ragged, x='A1', y='B1', agg=ds.count(), axis=1)
        ... tf.shade(agg)

        Instead of filling regions to the y=0 line, fill to the line with
        coordinates df_ragged.A1 by df_ragged.B2
        >>> agg = cvs.area(df_ragged, x='A1', y='B1', y_stack='B2', # doctest: +SKIP
        ...                agg=ds.count(), axis=1)
        ... tf.shade(agg)

        (requires pandas >= 0.24.0)
        """
        from .glyphs import (
            AreaToZeroAxis0, AreaToLineAxis0,
            AreaToZeroAxis0Multi, AreaToLineAxis0Multi,
            AreaToZeroAxis1, AreaToLineAxis1,
            AreaToZeroAxis1XConstant, AreaToLineAxis1XConstant,
            AreaToZeroAxis1YConstant, AreaToLineAxis1YConstant,
            AreaToZeroAxis1Ragged, AreaToLineAxis1Ragged,
        )
        from .reductions import any as any_rdn
        if agg is None:
            agg = any_rdn()

        # Broadcast column specifications to handle cases where
        # x is a list and y is a string or vice versa
        orig_x, orig_y, orig_y_stack = x, y, y_stack
        x, y, y_stack = _broadcast_column_specifications(x, y, y_stack)

        if axis == 0:
            if y_stack is None:
                if (isinstance(x, (Number, string_types)) and
                        isinstance(y, (Number, string_types))):
                    glyph = AreaToZeroAxis0(x, y)
                elif (isinstance(x, (list, tuple)) and
                      isinstance(y, (list, tuple))):
                    glyph = AreaToZeroAxis0Multi(tuple(x), tuple(y))
                else:
                    raise ValueError("""
Invalid combination of x and y arguments to Canvas.area when axis=0.
    Received:
        x: {x}
        y: {y}
See docstring for more information on valid usage""".format(
                        x=repr(x), y=repr(y)))
            else:
                # y_stack is not None
                if (isinstance(x, (Number, string_types)) and
                        isinstance(y, (Number, string_types)) and
                        isinstance(y_stack, (Number, string_types))):

                    glyph = AreaToLineAxis0(x, y, y_stack)
                elif (isinstance(x, (list, tuple)) and
                      isinstance(y, (list, tuple)) and
                      isinstance(y_stack, (list, tuple))):
                    glyph = AreaToLineAxis0Multi(
                        tuple(x), tuple(y), tuple(y_stack))
                else:
                    raise ValueError("""
Invalid combination of x, y, and y_stack arguments to Canvas.area when axis=0.
    Received:
        x: {x}
        y: {y}
        y_stack: {y_stack}
See docstring for more information on valid usage""".format(
                        x=repr(orig_x),
                        y=repr(orig_y),
                        y_stack=repr(orig_y_stack)))

        elif axis == 1:
            if y_stack is None:
                if (isinstance(x, (list, tuple)) and
                        isinstance(y, (list, tuple))):
                    glyph = AreaToZeroAxis1(tuple(x), tuple(y))
                elif (isinstance(x, np.ndarray) and
                      isinstance(y, (list, tuple))):
                    glyph = AreaToZeroAxis1XConstant(x, tuple(y))
                elif (isinstance(x, (list, tuple)) and
                      isinstance(y, np.ndarray)):
                    glyph = AreaToZeroAxis1YConstant(tuple(x), y)
                elif (isinstance(x, (Number, string_types)) and
                      isinstance(y, (Number, string_types))):
                    glyph = AreaToZeroAxis1Ragged(x, y)
                else:
                    raise ValueError("""
Invalid combination of x and y arguments to Canvas.area when axis=1.
    Received:
        x: {x}
        y: {y}
See docstring for more information on valid usage""".format(
                        x=repr(x), y=repr(y)))
            else:
                if (isinstance(x, (list, tuple)) and
                        isinstance(y, (list, tuple)) and
                        isinstance(y_stack, (list, tuple))):
                    glyph = AreaToLineAxis1(
                        tuple(x), tuple(y), tuple(y_stack))
                elif (isinstance(x, np.ndarray) and
                      isinstance(y, (list, tuple)) and
                      isinstance(y_stack, (list, tuple))):
                    glyph = AreaToLineAxis1XConstant(
                        x, tuple(y), tuple(y_stack))
                elif (isinstance(x, (list, tuple)) and
                      isinstance(y, np.ndarray) and
                      isinstance(y_stack, np.ndarray)):
                    glyph = AreaToLineAxis1YConstant(tuple(x), y, y_stack)
                elif (isinstance(x, (Number, string_types)) and
                      isinstance(y, (Number, string_types)) and
                      isinstance(y_stack, (Number, string_types))):
                    glyph = AreaToLineAxis1Ragged(x, y, y_stack)
                else:
                    raise ValueError("""
Invalid combination of x, y, and y_stack arguments to Canvas.area when axis=1.
    Received:
        x: {x}
        y: {y}
        y_stack: {y_stack}
See docstring for more information on valid usage""".format(
                        x=repr(orig_x),
                        y=repr(orig_y),
                        y_stack=repr(orig_y_stack)))
        else:
            raise ValueError("""
The axis argument to Canvas.line must be 0 or 1
    Received: {axis}""".format(axis=axis))

        return bypixel(source, self, glyph, agg)

    def polygons(self, source, geometry, agg=None):
        """Compute a reduction by pixel, mapping data to pixels as one or
        more filled polygons.

        Parameters
        ----------
        source : xarray.DataArray or Dataset
            The input datasource.
        geometry : str
            Column name of a PolygonsArray of the coordinates of each line.
        agg : Reduction, optional
            Reduction to compute. Default is ``any()``.

        Returns
        -------
        data : xarray.DataArray

        Examples
        --------
        >>> import datashader as ds  # doctest: +SKIP
        ... import datashader.transfer_functions as tf
        ... from spatialpandas.geometry import PolygonArray
        ... from spatialpandas import GeoDataFrame
        ... import pandas as pd
        ...
        ... polygons = PolygonArray([
        ...     # First Element
        ...     [[0, 0, 1, 0, 2, 2, -1, 4, 0, 0],  # Filled quadrilateral (CCW order)
        ...      [0.5, 1,  1, 2,  1.5, 1.5,  0.5, 1],     # Triangular hole (CW order)
        ...      [0, 2, 0, 2.5, 0.5, 2.5, 0.5, 2, 0, 2],  # Rectangular hole (CW order)
        ...      [2.5, 3, 3.5, 3, 3.5, 4, 2.5, 3],  # Filled triangle
        ...     ],
        ...
        ...     # Second Element
        ...     [[3, 0, 3, 2, 4, 2, 4, 0, 3, 0],  # Filled rectangle (CCW order)
        ...      # Rectangular hole (CW order)
        ...      [3.25, 0.25, 3.75, 0.25, 3.75, 1.75, 3.25, 1.75, 3.25, 0.25],
        ...     ]
        ... ])
        ...
        ... df = GeoDataFrame({'polygons': polygons, 'v': range(len(polygons))})
        ...
        ... cvs = ds.Canvas()
        ... agg = cvs.polygons(df, geometry='polygons', agg=ds.sum('v'))
        ... tf.shade(agg)
        """
        from .glyphs import PolygonGeom
        from .reductions import any as any_rdn
        from spatialpandas import GeoDataFrame
        from spatialpandas.dask import DaskGeoDataFrame
        if isinstance(source, DaskGeoDataFrame):
            # Downselect partitions to those that may contain polygons in viewport
            x_range = self.x_range if self.x_range is not None else (None, None)
            y_range = self.y_range if self.y_range is not None else (None, None)
            source = source.cx_partitions[slice(*x_range), slice(*y_range)]
        elif not isinstance(source, GeoDataFrame):
            raise ValueError(
                "source must be an instance of spatialpandas.GeoDataFrame or \n"
                "spatialpandas.dask.DaskGeoDataFrame.\n"
                "  Received value of type {typ}".format(typ=type(source)))

        if agg is None:
            agg = any_rdn()
        glyph = PolygonGeom(geometry)
        return bypixel(source, self, glyph, agg)

    def quadmesh(self, source, x=None, y=None, agg=None):
        """Samples a recti- or curvi-linear quadmesh by canvas size and bounds.
        Parameters
        ----------
        source : xarray.DataArray or Dataset
            The input datasource.
        x, y : str
            Column names for the x and y coordinates of each point.
        agg : Reduction, optional
            Reduction to compute. Default is ``mean()``. Note that agg is ignored when
            upsampling.
        Returns
        -------
        data : xarray.DataArray
        """
        from .glyphs import QuadMeshRaster, QuadMeshRectilinear, QuadMeshCurvilinear

        # Determine reduction operation
        from .reductions import mean as mean_rnd

        if isinstance(source, Dataset):
            if agg is None or agg.column is None:
                name = list(source.data_vars)[0]
            else:
                name = agg.column
            # Keep as dataset so that source[agg.column] works
            source = source[[name]]
        elif isinstance(source, DataArray):
            # Make dataset so that source[agg.column] works
            name = source.name
            source = source.to_dataset()
        else:
            raise ValueError("Invalid input type")

        if agg is None:
            agg = mean_rnd(name)

        if x is None and y is None:
            y, x = source[name].dims
        elif not x or not y:
            raise ValueError("Either specify both x and y coordinates"
                             "or allow them to be inferred.")
        yarr, xarr = source[y], source[x]

        if (yarr.ndim > 1 or xarr.ndim > 1) and xarr.dims != yarr.dims:
            raise ValueError("Ensure that x- and y-coordinate arrays "
                             "share the same dimensions. x-coordinates "
                             "are indexed by %s dims while "
                             "y-coordinates are indexed by %s dims." %
                             (xarr.dims, yarr.dims))

        if (name is not None
                and agg.column is not None
                and agg.column != name):
            raise ValueError('DataArray name %r does not match '
                             'supplied reduction %s.' %
                             (source.name, agg))

        if xarr.ndim == 1:
            xaxis_linear = self.x_axis is _axis_lookup["linear"]
            yaxis_linear = self.y_axis is _axis_lookup["linear"]
            even_yspacing = np.allclose(
                yarr, np.linspace(yarr[0], yarr[-1], len(yarr))
            )
            even_xspacing = np.allclose(
                xarr, np.linspace(xarr[0], xarr[-1], len(xarr))
            )

            if xaxis_linear and yaxis_linear and even_xspacing and even_yspacing:
                # Source is a raster, where all x and y coordinates are evenly spaced
                glyph = QuadMeshRaster(x, y, name)
                upsample_width, upsample_height = glyph.is_upsample(
                        source, x, y, name, self.x_range, self.y_range,
                        self.plot_width, self.plot_height
                )
                if upsample_width and upsample_height:
                    # Override aggregate with more efficient one for upsampling
                    agg = rd._upsample(name)
                    return bypixel(source, self, glyph, agg)
                elif not upsample_width and not upsample_height:
                    # Downsample both width and height
                    return bypixel(source, self, glyph, agg)
                else:
                    # Mix of upsampling and downsampling
                    # Use general rectilinear quadmesh implementation
                    glyph = QuadMeshRectilinear(x, y, name)
                    return bypixel(source, self, glyph, agg)
            else:
                # Source is a general rectilinear quadmesh
                glyph = QuadMeshRectilinear(x, y, name)
                return bypixel(source, self, glyph, agg)
        elif xarr.ndim == 2:
            glyph = QuadMeshCurvilinear(x, y, name)
            return bypixel(source, self, glyph, agg)
        else:
            raise ValueError("""\
x- and y-coordinate arrays must have 1 or 2 dimensions.
    Received arrays with dimensions: {dims}""".format(
                dims=list(xarr.dims)))

    # TODO re 'untested', below: Consider replacing with e.g. a 3x3
    # array in the call to Canvas (plot_height=3,plot_width=3), then
    # show the output as a numpy array that has a compact
    # representation
    def trimesh(self, vertices, simplices, mesh=None, agg=None, interp=True, interpolate=None):
        """Compute a reduction by pixel, mapping data to pixels as a triangle.

        >>> import datashader as ds
        >>> verts = pd.DataFrame({'x': [0, 5, 10],
        ...                         'y': [0, 10, 0],
        ...                         'weight': [1, 5, 3]},
        ...                        columns=['x', 'y', 'weight'])
        >>> tris = pd.DataFrame({'v0': [2], 'v1': [0], 'v2': [1]},
        ...                       columns=['v0', 'v1', 'v2'])
        >>> cvs = ds.Canvas(x_range=(verts.x.min(), verts.x.max()),
        ...                 y_range=(verts.y.min(), verts.y.max()))
        >>> untested = cvs.trimesh(verts, tris)

        Parameters
        ----------
        vertices : pandas.DataFrame, dask.DataFrame
            The input datasource for triangle vertex coordinates. These can be
            interpreted as the x/y coordinates of the vertices, with optional
            weights for value interpolation. Columns should be ordered
            corresponding to 'x', 'y', followed by zero or more (optional)
            columns containing vertex values. The rows need not be ordered.
            The column data types must be floating point or integer.
        simplices : pandas.DataFrame, dask.DataFrame
            The input datasource for triangle (simplex) definitions. These can
            be interpreted as rows of ``vertices``, aka positions in the
            ``vertices`` index. Columns should be ordered corresponding to
            'vertex0', 'vertex1', and 'vertex2'. Order of the vertices can be
            clockwise or counter-clockwise; it does not matter as long as the
            data is consistent for all simplices in the dataframe. The
            rows need not be ordered.  The data type for the first
            three columns in the dataframe must be integer.
        agg : Reduction, optional
            Reduction to compute. Default is ``mean()``.
        mesh : pandas.DataFrame, optional
            An ordered triangle mesh in tabular form, used for optimization
            purposes. This dataframe is expected to have come from
            ``datashader.utils.mesh()``. If this argument is not None, the first
            two arguments are ignored.
        interpolate : str, optional default=linear
            Method to use for interpolation between specified values. ``nearest``
            means to use a single value for the whole triangle, and ``linear``
            means to do bilinear interpolation of the pixels within each
            triangle (a weighted average of the vertex values). For
            backwards compatibility, also accepts ``interp=True`` for ``linear``
            and ``interp=False`` for ``nearest``.
        """
        from .glyphs import Triangles
        from .reductions import mean as mean_rdn
        from .utils import mesh as create_mesh

        source = mesh

        # 'interp' argument is deprecated as of datashader=0.6.4
        if interpolate is not None:
            if interpolate == 'linear':
                interp = True
            elif interpolate == 'nearest':
                interp = False
            else:
                raise ValueError('Invalid interpolate method: options include {}'.format(['linear','nearest']))

        # Validation is done inside the [pd]d_mesh utility functions
        if source is None:
            source = create_mesh(vertices, simplices)

        verts_have_weights = len(vertices.columns) > 2
        if verts_have_weights:
            weight_col = vertices.columns[2]
        else:
            weight_col = simplices.columns[3]

        if agg is None:
            agg = mean_rdn(weight_col)
        elif agg.column is None:
            agg.column = weight_col

        cols = source.columns
        x, y, weights = cols[0], cols[1], cols[2:]

        return bypixel(source, self, Triangles(x, y, weights, weight_type=verts_have_weights, interp=interp), agg)

    def raster(self,
               source,
               layer=None,
               upsample_method='linear',    # Deprecated as of datashader=0.6.4
               downsample_method=rd.mean(), # Deprecated as of datashader=0.6.4
               nan_value=None,
               agg=None,
               interpolate=None,
               chunksize=None,
               max_mem=None):
        """Sample a raster dataset by canvas size and bounds.

        Handles 2D or 3D xarray DataArrays, assuming that the last two
        array dimensions are the y- and x-axis that are to be
        resampled. If a 3D array is supplied a layer may be specified
        to resample to select the layer along the first dimension to
        resample.

        Missing values (those having the value indicated by the
        "nodata" attribute of the raster) are replaced with `NaN` if
        floats, and 0 if int.

        Also supports resampling out-of-core DataArrays backed by dask
        Arrays. By default it will try to maintain the same chunksize
        in the output array but a custom chunksize may be provided.
        If there are memory constraints they may be defined using the
        max_mem parameter, which determines how large the chunks in
        memory may be.

        Parameters
        ----------
        source : xarray.DataArray or xr.Dataset
            2D or 3D labelled array (if Dataset, the agg reduction must
            define the data variable).
        layer : float
            For a 3D array, value along the z dimension : optional default=None
        ds_method : str (optional)
            Grid cell aggregation method for a possible downsampling.
        us_method : str (optional)
            Grid cell interpolation method for a possible upsampling.
        nan_value : int or float, optional
            Optional nan_value which will be masked out when applying
            the resampling.
        agg : Reduction, optional default=mean()
            Resampling mode when downsampling raster. The supported
            options include: first, last, mean, mode, var, std, min,
            The agg can be specified as either a string name or as a
            reduction function, but note that the function object will
            be used only to extract the agg type (mean, max, etc.) and
            the optional column name; the hardcoded raster code
            supports only a fixed set of reductions and ignores the
            actual code of the provided agg.
        interpolate : str, optional  default=linear
            Resampling mode when upsampling raster.
            options include: nearest, linear.
        chunksize : tuple(int, int) (optional)
            Size of the output chunks. By default this the chunk size is
            inherited from the *src* array.
        max_mem : int (optional)
            The maximum number of bytes that should be loaded into memory
            during the regridding operation.

        Returns
        -------
        data : xarray.Dataset
        """
        # For backwards compatibility
        if agg         is None: agg=downsample_method
        if interpolate is None: interpolate=upsample_method

        upsample_methods = ['nearest','linear']

        downsample_methods = {'first':'first', rd.first:'first',
                              'last':'last',   rd.last:'last',
                              'mode':'mode',   rd.mode:'mode',
                              'mean':'mean',   rd.mean:'mean',
                              'var':'var',     rd.var:'var',
                              'std':'std',     rd.std:'std',
                              'min':'min',     rd.min:'min',
                              'max':'max',     rd.max:'max'}

        if interpolate not in upsample_methods:
            raise ValueError('Invalid interpolate method: options include {}'.format(upsample_methods))

        if not isinstance(source, (DataArray, Dataset)):
            raise ValueError('Expected xarray DataArray or Dataset as '
                             'the data source, found %s.'
                             % type(source).__name__)

        column = None
        if isinstance(agg, rd.Reduction):
            agg, column = type(agg), agg.column
            if (isinstance(source, DataArray) and column is not None
                and source.name != column):
                agg_repr = '%s(%r)' % (agg.__name__, column)
                raise ValueError('DataArray name %r does not match '
                                 'supplied reduction %s.' %
                                 (source.name, agg_repr))

        if isinstance(source, Dataset):
            data_vars = list(source.data_vars)
            if column is None:
                raise ValueError('When supplying a Dataset the agg reduction '
                                 'must specify the variable to aggregate. '
                                 'Available data_vars include: %r.' % data_vars)
            elif column not in source.data_vars:
                raise KeyError('Supplied reduction column %r not found '
                               'in Dataset, expected one of the following '
                               'data variables: %r.' % (column, data_vars))
            source = source[column]

        if agg not in downsample_methods.keys():
            raise ValueError('Invalid aggregation method: options include {}'.format(list(downsample_methods.keys())))
        ds_method = downsample_methods[agg]

        if source.ndim not in [2, 3]:
            raise ValueError('Raster aggregation expects a 2D or 3D '
                             'DataArray, found %s dimensions' % source.ndim)

        res = calc_res(source)
        ydim, xdim = source.dims[-2:]
        xvals, yvals = source[xdim].values, source[ydim].values
        left, bottom, right, top = calc_bbox(xvals, yvals, res)
        if layer is not None:
            source=source.sel(**{source.dims[0]: layer})
        array = orient_array(source, res)
        dtype = array.dtype

        if nan_value is not None:
            mask = array==nan_value
            array = np.ma.masked_array(array, mask=mask, fill_value=nan_value)
            fill_value = nan_value
        else:
            fill_value = np.NaN

        if self.x_range is None: self.x_range = (left,right)
        if self.y_range is None: self.y_range = (bottom,top)

        # window coordinates
        xmin = max(self.x_range[0], left)
        ymin = max(self.y_range[0], bottom)
        xmax = min(self.x_range[1], right)
        ymax = min(self.y_range[1], top)

        width_ratio = min((xmax - xmin) / (self.x_range[1] - self.x_range[0]), 1)
        height_ratio = min((ymax - ymin) / (self.y_range[1] - self.y_range[0]), 1)

        if np.isclose(width_ratio, 0) or np.isclose(height_ratio, 0):
            raise ValueError('Canvas x_range or y_range values do not match closely enough with the data source to be able to accurately rasterize. Please provide ranges that are more accurate.')

        w = max(int(round(self.plot_width * width_ratio)), 1)
        h = max(int(round(self.plot_height * height_ratio)), 1)
        cmin, cmax = get_indices(xmin, xmax, xvals, res[0])
        rmin, rmax = get_indices(ymin, ymax, yvals, res[1])

        kwargs = dict(w=w, h=h, ds_method=ds_method,
                      us_method=interpolate, fill_value=fill_value)
        if array.ndim == 2:
            source_window = array[rmin:rmax+1, cmin:cmax+1]
            if ds_method in ['var', 'std']:
                source_window = source_window.astype('f')
            if isinstance(source_window, da.Array):
                data = resample_2d_distributed(
                    source_window, chunksize=chunksize, max_mem=max_mem,
                    **kwargs)
            else:
                data = resample_2d(source_window, **kwargs)
            layers = 1
        else:
            source_window = array[:, rmin:rmax+1, cmin:cmax+1]
            if ds_method in ['var', 'std']:
                source_window = source_window.astype('f')
            arrays = []
            for arr in source_window:
                if isinstance(arr, da.Array):
                    arr = resample_2d_distributed(
                        arr, chunksize=chunksize, max_mem=max_mem,
                        **kwargs)
                else:
                    arr = resample_2d(arr, **kwargs)
                arrays.append(arr)
            data = np.dstack(arrays)
            layers = len(arrays)

        if w != self.plot_width or h != self.plot_height:
            num_height = self.plot_height - h
            num_width = self.plot_width - w

            lpad = xmin - self.x_range[0]
            rpad = self.x_range[1] - xmax
            lpct = lpad / (lpad + rpad) if lpad + rpad > 0 else 0
            left = max(int(np.ceil(num_width * lpct)), 0)
            right = max(num_width - left, 0)
            lshape, rshape = (self.plot_height, left), (self.plot_height, right)
            if layers > 1:
                lshape, rshape = lshape + (layers,), rshape + (layers,)
            left_pad = np.full(lshape, fill_value, source_window.dtype)
            right_pad = np.full(rshape, fill_value, source_window.dtype)

            tpad = ymin - self.y_range[0]
            bpad = self.y_range[1] - ymax
            tpct = tpad / (tpad + bpad) if tpad + bpad > 0 else 0
            top = max(int(np.ceil(num_height * tpct)), 0)
            bottom = max(num_height - top, 0)
            tshape, bshape = (top, w), (bottom, w)
            if layers > 1:
                tshape, bshape = tshape + (layers,), bshape + (layers,)
            top_pad = np.full(tshape, fill_value, source_window.dtype)
            bottom_pad = np.full(bshape, fill_value, source_window.dtype)

            concat = da.concatenate if isinstance(data, da.Array) else np.concatenate
            arrays = (top_pad, data) if top_pad.shape[0] > 0 else (data,)
            if bottom_pad.shape[0] > 0:
                arrays += (bottom_pad,)
            data = concat(arrays, axis=0) if len(arrays) > 1 else arrays[0]
            arrays = (left_pad, data) if left_pad.shape[1] > 0 else (data,)
            if right_pad.shape[1] > 0:
                arrays += (right_pad,)
            data = concat(arrays, axis=1) if len(arrays) > 1 else arrays[0]

        # Reorient array to original orientation
        if res[1] > 0: data = data[::-1]
        if res[0] < 0: data = data[:, ::-1]

        # Restore nan_value from masked array
        if nan_value is not None:
            data = data.filled()

        # Restore original dtype
        if dtype != data.dtype:
            data = data.astype(dtype)

        # Compute DataArray metadata
        xs, ys = compute_coords(self.plot_width, self.plot_height, self.x_range, self.y_range, res)
        coords = {xdim: xs, ydim: ys}
        dims = [ydim, xdim]
        attrs = dict(res=res[0])
        if source._file_obj is not None and hasattr(source._file_obj, 'nodata'):
            attrs['nodata'] = source._file_obj.nodata

        # Handle DataArray with layers
        if data.ndim == 3:
            data = data.transpose([2, 0, 1])
            layer_dim = source.dims[0]
            coords[layer_dim] = source.coords[layer_dim]
            dims = [layer_dim]+dims
        return DataArray(data, coords=coords, dims=dims, attrs=attrs)

    def validate(self):
        """Check that parameter settings are valid for this object"""
        self.x_axis.validate(self.x_range)
        self.y_axis.validate(self.y_range)


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

    # Convert 1D xarray DataArrays and DataSets into Dask DataFrames
    if isinstance(source, DataArray) and source.ndim == 1:
        if not source.name:
            source.name = 'value'
        source = source.reset_coords()
    if isinstance(source, Dataset) and len(source.dims) == 1:
        columns = list(source.coords.keys()) + list(source.data_vars.keys())
        cols_to_keep = _cols_to_keep(columns, glyph, agg)
        source = source.drop([col for col in columns if col not in cols_to_keep])
        source = source.to_dask_dataframe()

    if (isinstance(source, pd.DataFrame) or
            (cudf and isinstance(source, cudf.DataFrame))):
        # Avoid datashape.Categorical instantiation bottleneck
        # by only retaining the necessary columns:
        # https://github.com/bokeh/datashader/issues/396
        # Preserve column ordering without duplicates
        cols_to_keep = _cols_to_keep(source.columns, glyph, agg)
        if len(cols_to_keep) < len(source.columns):
            source = source[cols_to_keep]
        dshape = dshape_from_pandas(source)
    elif isinstance(source, dd.DataFrame):
        dshape = dshape_from_dask(source)
    elif isinstance(source, Dataset):
        # Multi-dimensional Dataset
        dshape = dshape_from_xarray_dataset(source)
    else:
        raise ValueError("source must be a pandas or dask DataFrame")
    schema = dshape.measure
    glyph.validate(schema)
    agg.validate(schema)
    canvas.validate()

    # All-NaN objects (e.g. chunks of arrays with no data) are valid in Datashader
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        return bypixel.pipeline(source, schema, canvas, glyph, agg)


def _cols_to_keep(columns, glyph, agg):
    cols_to_keep = OrderedDict({col: False for col in columns})
    for col in glyph.required_columns():
        cols_to_keep[col] = True

    if hasattr(agg, 'values'):
        for subagg in agg.values:
            if subagg.column is not None:
                cols_to_keep[subagg.column] = True
    elif hasattr(agg, 'columns'):
        for column in agg.columns:
            if column is not None:
                cols_to_keep[column] = True
    elif agg.column is not None:
        cols_to_keep[agg.column] = True
    return [col for col, keepit in cols_to_keep.items() if keepit]


def _broadcast_column_specifications(*args):
    lengths = {len(a) for a in args if isinstance(a, (list, tuple))}
    if len(lengths) != 1:
        # None of the inputs are lists/tuples, return them as is
        return args
    else:
        n = lengths.pop()
        return tuple(
            (arg,) * n if isinstance(arg, (Number, string_types)) else arg
            for arg in args
        )


bypixel.pipeline = Dispatcher()
