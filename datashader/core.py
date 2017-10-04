from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.array import Array
from xarray import DataArray
from collections import OrderedDict

from .utils import Dispatcher, ngjit, calc_res, calc_bbox, orient_array, compute_coords, get_indices, dshape_from_pandas, dshape_from_dask, categorical_in_dtypes
from .resampling import (resample_2d, US_NEAREST, US_LINEAR, DS_FIRST, DS_LAST,
                         DS_MEAN, DS_MODE, DS_VAR, DS_STD, DS_MIN, DS_MAX)


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
               layer=None,
               upsample_method='linear',
               downsample_method='mean',
               nan_value=None):
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
        source : xarray.DataArray
            input datasource most likely obtain from `xr.open_rasterio()`.
        layer : int
            source layer number : optional default=None
        upsample_method : str, optional default=linear
            resample mode when upsampling raster.
            options include: nearest, linear.
        downsample_method : str, optional default=mean
            resample mode when downsampling raster.
            options include: first, last, mean, mode, var, std
        nan_value : int or float, optional
            Optional nan_value which will be masked out when applying
            the resampling.

        Returns
        -------
        data : xarray.Dataset

        """
        upsample_methods = dict(nearest=US_NEAREST,
                                linear=US_LINEAR)

        downsample_methods = dict(first=DS_FIRST,
                                  last=DS_LAST,
                                  mean=DS_MEAN,
                                  mode=DS_MODE,
                                  var=DS_VAR,
                                  std=DS_STD,
                                  min=DS_MIN,
                                  max=DS_MAX)

        if upsample_method not in upsample_methods.keys():
            raise ValueError('Invalid upsample method: options include {}'.format(list(upsample_methods.keys())))
        if downsample_method not in downsample_methods.keys():
            raise ValueError('Invalid downsample method: options include {}'.format(list(downsample_methods.keys())))

        res = calc_res(source)
        ydim, xdim = source.dims[-2:]
        xvals, yvals = source[xdim].values, source[ydim].values
        left, bottom, right, top = calc_bbox(xvals, yvals, res)
        array = orient_array(source, res, layer)
        dtype = array.dtype

        if nan_value is not None:
            mask = array==nan_value
            array = np.ma.masked_array(array, mask=mask, fill_value=nan_value)
            fill_value = nan_value
        else:
            fill_value = np.NaN

        # window coordinates
        xmin = max(self.x_range[0], left)
        ymin = max(self.y_range[0], bottom)
        xmax = min(self.x_range[1], right)
        ymax = min(self.y_range[1], top)

        width_ratio = (xmax - xmin) / (self.x_range[1] - self.x_range[0])
        height_ratio = (ymax - ymin) / (self.y_range[1] - self.y_range[0])

        if np.isclose(width_ratio, 0) or np.isclose(height_ratio, 0):
            raise ValueError('Canvas x_range or y_range values do not match closely-enough with the data source to be able to accurately rasterize. Please provide ranges that are more accurate.')

        w = int(np.ceil(self.plot_width * width_ratio))
        h = int(np.ceil(self.plot_height * height_ratio))
        cmin, cmax = get_indices(xmin, xmax, xvals, res[0])
        rmin, rmax = get_indices(ymin, ymax, yvals, res[1])

        kwargs = dict(w=w, h=h, ds_method=downsample_methods[downsample_method],
                      us_method=upsample_methods[upsample_method], fill_value=fill_value)
        if array.ndim == 2:
            source_window = array[rmin:rmax+1, cmin:cmax+1]
            if isinstance(source_window, Array):
                source_window = source_window.compute()
            if downsample_method in ['var', 'std']:
                source_window = source_window.astype('f')
            data = resample_2d(source_window, **kwargs)
            layers = 1
        else:
            source_window = array[:, rmin:rmax+1, cmin:cmax+1]
            if downsample_method in ['var', 'std']:
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
            left = int(np.ceil(num_width * lpct))
            right = num_width - left
            lshape, rshape = (self.plot_height, left), (self.plot_height, right)
            if layers > 1:
                lshape, rshape = lshape + (layers,), rshape + (layers,)
            left_pad = np.full(lshape, fill_value, source_window.dtype)
            right_pad = np.full(rshape, fill_value, source_window.dtype)

            tpad = ymin - self.y_range[0]
            bpad = self.y_range[1] - ymax
            tpct = tpad / (tpad + bpad) if tpad + bpad > 0 else 0
            top = int(np.ceil(num_height * tpct))
            bottom = num_height - top
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
    # Avoid datashape.Categorical instantiation bottleneck
    # by only retaining the necessary columns:
    # https://github.com/bokeh/datashader/issues/396
    if categorical_in_dtypes(source.dtypes.values):
        # Preserve column ordering without duplicates
        cols_to_keep = OrderedDict({col: False for col in source.columns})
        cols_to_keep[glyph.x] = True
        cols_to_keep[glyph.y] = True
        if hasattr(agg, 'values'):
            for subagg in agg.values:
                if subagg.column is not None:
                    cols_to_keep[subagg.column] = True
        elif agg.column is not None:
            cols_to_keep[agg.column] = True
        src = source[[col for col, keepit in cols_to_keep.items() if keepit]]
    else:
        src = source

    if isinstance(src, pd.DataFrame):
        dshape = dshape_from_pandas(src)
    elif isinstance(src, dd.DataFrame):
        dshape = dshape_from_dask(src)
    else:
        raise ValueError("source must be a pandas or dask DataFrame")
    schema = dshape.measure
    glyph.validate(schema)
    agg.validate(schema)
    canvas.validate()
    return bypixel.pipeline(source, schema, canvas, glyph, agg)


bypixel.pipeline = Dispatcher()
