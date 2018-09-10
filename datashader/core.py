from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.array import Array
from xarray import DataArray, Dataset
from collections import OrderedDict

from .utils import Dispatcher, ngjit, calc_res, calc_bbox, orient_array, compute_coords
from .utils import get_indices, dshape_from_pandas, dshape_from_dask
from .utils import Expr # noqa (API import)
from .resampling import resample_2d
from . import reductions as rd


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
        return np.log10(val)

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
        from .reductions import count as count_rdn
        if agg is None:
            agg = count_rdn()
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
        from .reductions import any as any_rdn
        if agg is None:
            agg = any_rdn()
        return bypixel(source, self, Line(x, y), agg)

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
               interpolate=None):
        """Sample a raster dataset by canvas size and bounds.

        Handles 2D or 3D xarray DataArrays, assuming that the last two
        array dimensions are the y- and x-axis that are to be
        resampled. If a 3D array is supplied a layer may be specified
        to resample to select the layer along the first dimension to
        resample.

        Missing values (those having the value indicated by the
        "nodata" attribute of the raster) are replaced with `NaN` if
        floats, and 0 if int.

        Parameters
        ----------
        source : xarray.DataArray or xr.Dataset
            2D or 3D labelled array (if Dataset, the agg reduction must
            define the data variable).
        layer : float
            For a 3D array, value along the z dimension : optional default=None
        interpolate : str, optional  default=linear
            Resampling mode when upsampling raster.
            options include: nearest, linear.
        agg : Reduction, optional default=mean()
            Resampling mode when downsampling raster.
            options include: first, last, mean, mode, var, std, min, max
            Also accepts string names, for backwards compatibility.
        nan_value : int or float, optional
            Optional nan_value which will be masked out when applying
            the resampling.

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
            if isinstance(source_window, Array):
                source_window = source_window.compute()
            if ds_method in ['var', 'std']:
                source_window = source_window.astype('f')
            data = resample_2d(source_window, **kwargs)
            layers = 1
        else:
            source_window = array[:, rmin:rmax+1, cmin:cmax+1]
            if ds_method in ['var', 'std']:
                source_window = source_window.astype('f')
            arrays = []
            for arr in source_window:
                if isinstance(arr, Array):
                    arr = arr.compute()
                arrays.append(resample_2d(arr, **kwargs))
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

            data = np.concatenate((top_pad, data, bottom_pad), axis=0)
            data = np.concatenate((left_pad, data, right_pad), axis=1)

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
        if source._file_obj is not None:
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
    if isinstance(source, pd.DataFrame):
        # Avoid datashape.Categorical instantiation bottleneck
        # by only retaining the necessary columns:
        # https://github.com/bokeh/datashader/issues/396
        # Preserve column ordering without duplicates
        cols_to_keep = OrderedDict({col: False for col in source.columns})
        cols_to_keep[glyph.x] = True
        cols_to_keep[glyph.y] = True
        if hasattr(glyph, 'z'):
            cols_to_keep[glyph.z[0]] = True
        if hasattr(agg, 'values'):
            for subagg in agg.values:
                if subagg.column is not None:
                    cols_to_keep[subagg.column] = True
        elif agg.column is not None:
            cols_to_keep[agg.column] = True
        cols_to_keep = [col for col, keepit in cols_to_keep.items() if keepit]
        if len(cols_to_keep) < len(source.columns):
            source = source[cols_to_keep]
        dshape = dshape_from_pandas(source)
    elif isinstance(source, dd.DataFrame):
        dshape = dshape_from_dask(source)
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


bypixel.pipeline = Dispatcher()
