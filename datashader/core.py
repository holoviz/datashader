from __future__ import absolute_import, division, print_function

import numpy as np
from datashape.predicates import istabular
from odo import discover
from xarray import DataArray

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
    """

    def compute_scale_and_translate(self, range, n):
        """Compute the scale and translate parameters for a linear transformation
        ``output = s * input + t``, mapping from data space to axis space.

        Parameters
        ----------
        range : tuple
            A tuple representing the range ``[min, max]`` along the axis, in
            data space. min is inclusive and max is exclusive.
        n : int
            The number of bins along the axis.

        Returns
        -------
        s, t : floats
            Parameters represe

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
        px = np.arange(n)
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
        return np.log10(val)

    @staticmethod
    @ngjit
    def inverse_mapper(val):
        return 10**val

    def validate(self, range):
        low, high = map(self.mapper, range)
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError('Range values must be >0 for a LogAxis')


_axis_lookup = {'linear': LinearAxis(), 'log': LogAxis()}


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

    def raster(self,
               source,
               band=1,
               resample_method='bilinear',
               use_overviews=True):
        """Sample a raster dataset by canvas size and bounds. Note: requires
        `xarray`, `rasterio`, and `scikit-image`.  Missing values (those having
        the value indicated by the "nodata" attribute of the raster) are
        replaced with `NaN` if floats, and 0 if int.

        Parameters
        ----------
        source : xarray.DataArray
            input datasource most likely obtain from `xarray.open_rasterio()`.
        band : int
            source band number : optional default=1
        resample_method : str, optional default=bilinear
            resample mode when resizing raster.
            options include: nearest, bilinear.
        use_overviews : bool, optional default=True
            flag to indicate whether to use overviews or use native resolution

        Returns
        -------
        data : xarray.Dataset

        Notes
        -------
        requires `xarray`, `rasterio`, and `scikit-image`.
        """

        try:
            import rasterio as rio
            from skimage.transform import resize
        except ImportError:
            raise ImportError('install rasterio and skimage to use this feature')

        resample_methods = dict(nearest=0, bilinear=1)

        if resample_method not in resample_methods.keys():
            raise ValueError('Invalid resample method: options include {}'.format(list(resample_methods.keys())))

        # setup output array
        full_data = np.empty(shape=(self.plot_width, self.plot_height)).astype(source.dtype)
        full_xs = np.linspace(self.x_range[0], self.x_range[1], self.plot_width)
        full_ys = np.linspace(self.y_range[0], self.y_range[1], self.plot_height)
        attrs = dict(res=source._file_obj.res[0], nodata=source._file_obj.nodata)
        full_arr = DataArray(full_data,
                             coords=[('x', full_xs), ('y', full_ys)],
                             attrs=attrs)

        # handle out-of-bounds case
        if (self.x_range[0] >= source._file_obj.bounds.right or
            self.x_range[1] <= source._file_obj.bounds.left or
            self.y_range[0] >= source._file_obj.bounds.top or
            self.y_range[1] <= source._file_obj.bounds.bottom):
            return full_arr

        # window coodinates
        xmin = max(self.x_range[0], source._file_obj.bounds.left)
        ymin = max(self.y_range[0], source._file_obj.bounds.bottom)
        xmax = min(self.x_range[1], source._file_obj.bounds.right)
        ymax = min(self.y_range[1], source._file_obj.bounds.top)

        width_ratio = (xmax - xmin) / (self.x_range[1] - self.x_range[0])
        height_ratio = (ymax - ymin) / (self.y_range[1] - self.y_range[0])

        w = int(np.ceil(self.plot_width * width_ratio))
        h = int(np.ceil(self.plot_height * height_ratio))

        rmin, cmin = source._file_obj.index(xmin, ymin)
        rmax, cmax = source._file_obj.index(xmax, ymax)

        if use_overviews:
            data = np.empty(shape=(h, w)).astype(source.dtype)
            data = source._file_obj.read(band, out=data, window=((rmax, rmin), (cmin, cmax)))
        else:
            data = source._file_obj.read(band, window=((rmax, rmin), (cmin, cmax)))

        is_int = np.issubdtype(data.dtype, np.integer)
        data[data == np.array(source._file_obj.nodata)] = 0 if is_int else np.nan

        # TODO: this resize should go away once rasterio has overview resample
        data = resize(data,
                      (h, w),
                      order=resample_methods[resample_method],
                      preserve_range=True)

        if w != self.plot_width or h != self.plot_height:
            num_height = self.plot_height - h
            num_width = self.plot_width - w

            lpad = xmin - self.x_range[0]
            rpad = self.x_range[1] - xmax
            lpct = lpad / (lpad + rpad) if lpad + rpad > 0 else 0
            left = int(np.ceil(num_width * lpct))
            right = num_width - left
            left_pad = np.empty(shape=(self.plot_height, left)).astype(source.dtype) * np.nan
            right_pad = np.empty(shape=(self.plot_height, right)).astype(source.dtype) * np.nan

            tpad = ymin - self.y_range[0]
            bpad = self.y_range[1] - ymax
            tpct = tpad / (tpad + bpad) if tpad + bpad > 0 else 0
            top = int(np.ceil(num_height * tpct))
            bottom = num_height - top
            top_pad = np.empty(shape=(top, w)).astype(source.dtype) * np.nan
            bottom_pad = np.empty(shape=(bottom, w)).astype(source.dtype) * np.nan

            data = np.concatenate((bottom_pad, data, top_pad), axis=0)
            data = np.concatenate((left_pad, data, right_pad), axis=1)

        data = np.flipud(data)
        attrs = dict(res=source._file_obj.res[0], nodata=source._file_obj.nodata)
        return DataArray(data,
                         dims=['x', 'y'],
                         attrs=attrs)

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
    dshape = discover(source)
    if not istabular(dshape):
        raise ValueError("source must be tabular")
    schema = dshape.measure
    glyph.validate(schema)
    agg.validate(schema)
    canvas.validate()
    return bypixel.pipeline(source, schema, canvas, glyph, agg)


bypixel.pipeline = Dispatcher()
