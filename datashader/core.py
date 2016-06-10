from __future__ import absolute_import, division, print_function

import numpy as np
from datashape.predicates import istabular
from odo import discover
from xarray import DataArray, align

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

    def raster(self,
               source,
               band=1,
               resample_method='bilinear',
               use_overviews=False,
               missing=None):
        """Sample a raster dataset by canvas size and bounds. Note: requires
        `rasterio` and `scikit-image`.

        Parameters
        ----------
        source : rasterio.Dataset
            input datasource most likely obtain from `rasterio.open()`.
        band : int
            source band number : optional default=1
        missing : number, optional
            Missing flag, default is `None` and missing values are replaced with `NaN`
            if floats, and 0 if int.
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
        requires `rasterio` and `scikit-image`.
        """

        try:
            import rasterio as rio
            from skimage.transform import resize
            from affine import Affine
        except ImportError:
            raise ImportError('install rasterio and skimage to use this feature')

        resample_methods = dict(nearest=0, bilinear=1)

        if resample_method not in resample_methods.keys():
            raise ValueError('Invalid resample method: options include {}'.format(list(resample_methods.keys())))

        # setup output array

        full_data = np.empty(shape=(self.plot_width, self.plot_height)).astype(source.profile['dtype'])
        full_xs = np.linspace(self.x_range[0], self.x_range[1], self.plot_width)
        full_ys = np.linspace(self.y_range[0], self.y_range[1], self.plot_height)
        attrs = dict(res=source.res[0], nodata=source.nodata)
        full_arr =  DataArray(full_data, 
                              coords=[('x', full_xs), ('y', full_ys)], 
                              attrs=attrs)

        # handle out-of-bounds case
        if (self.x_range[0] >= source.bounds.right or
            self.x_range[1] <= source.bounds.left or
            self.y_range[0] >= source.bounds.top or
            self.y_range[1] <= source.bounds.bottom):
            return full_arr

        # window coodinates
        xmin = max(self.x_range[0], source.bounds.left)
        ymin = max(self.y_range[0], source.bounds.bottom)
        xmax = min(self.x_range[1], source.bounds.right)
        ymax = min(self.y_range[1], source.bounds.top)

        width_ratio = (xmax - xmin) / (self.x_range[1] - self.x_range[0])
        height_ratio = (ymax - ymin) / (self.y_range[1] - self.y_range[0])

        w = int(np.ceil(self.plot_width * width_ratio))
        h = int(np.ceil(self.plot_height * height_ratio))

        rmin, cmin = source.index(xmin, ymin)
        rmax, cmax = source.index(xmax, ymax)

        if use_overviews:
            data = np.empty(shape=(w, h)).astype(source.profile['dtype'])
            data = source.read(band, out=data, window=((rmax, rmin), (cmin, cmax)))
        else:
            data = source.read(band, window=((rmax, rmin), (cmin, cmax)))

        if missing and source.nodata:
            data[data == source.nodata] = missing
        elif source.nodata:
            data[data == source.nodata] = 0 if 'i' in data.dtype.str else np.nan
        else:
            print('warning, rasterio source does not indicate nodata value')

        # TODO: this resize should go away once rasterio has overview resample
        window_data = resize(np.flipud(data),
                      (w, h),
                      order=resample_methods[resample_method],
                      preserve_range=True)

        attrs = dict(res=source.res[0], nodata=source.nodata)
        return DataArray(data,
                         dims=['x', 'y'],
                         attrs=attrs)

        if w != self.plot_width or h != self.plot_height:
            import pdb; pdb.set_trace()
            x_res = self.x_range[1] - self.x_range[0] / self.plot_width 
            y_res = self.y_range[1] - self.y_range[0] / self.plot_height 
            transform = (self.x_range[0], x_res, 0.0, self.y_range[0], 0.0, y_res)
            affine_tranform = ~Affine.from_gdal(*transform)
            fxmin, fymin = (xmin, ymin) * affine_transform
            fxmax, fymax = (xmax, ymax) * affine_transform
            full_arr.data[int(fxmin):int(fxmax), int(fymin):int(fymax)] = window_data
            return full_arr

        else:
            attrs = dict(res=source.res[0], nodata=source.nodata)
            return DataArray(data,
                             dims=['x', 'y'],
                             attrs=attrs)

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
