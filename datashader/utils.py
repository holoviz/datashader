from __future__ import absolute_import, division, print_function

import os

from inspect import getmro

import numba as nb
import numpy as np

from xarray import DataArray

from datashape import Unit
from datashape.predicates import launder
from datashape.typesets import real

ngjit = nb.jit(nopython=True, nogil=True)


class Dispatcher(object):
    """Simple single dispatch."""
    def __init__(self):
        self._lookup = {}

    def register(self, type, func=None):
        """Register dispatch of `func` on arguments of type `type`"""
        if func is None:
            return lambda f: self.register(type, f)
        if isinstance(type, tuple):
            for t in type:
                self.register(t, func)
        else:
            self._lookup[type] = func
        return func

    def __call__(self, head, *rest, **kwargs):
        # We dispatch first on type(head), and fall back to iterating through
        # the mro. This is significantly faster in the common case where
        # type(head) is in the lookup, with only a small penalty on fall back.
        lk = self._lookup
        typ = type(head)
        if typ in lk:
            return lk[typ](head, *rest, **kwargs)
        for cls in getmro(typ)[1:]:
            if cls in lk:
                return lk[cls](head, *rest, **kwargs)
        raise TypeError("No dispatch for {0} type".format(typ))


def isreal(dt):
    """Check if a datashape is numeric and real.

    Example
    -------
    >>> isreal('int32')
    True
    >>> isreal('float64')
    True
    >>> isreal('string')
    False
    >>> isreal('complex64')
    False
    """
    dt = launder(dt)
    return isinstance(dt, Unit) and dt in real


def calc_res(raster):
    """Calculate the resolution of xarray.DataArray raster and return it as the
    two-tuple (xres, yres).
    """
    w, h = raster.shape[2], raster.shape[1]
    xres = (raster.x.values[w-1] - raster.x.values[0]) / (w - 1)
    yres = (raster.y.values[0] - raster.y.values[h-1]) / (h - 1)
    return xres, yres

def calc_bbox(xs, ys, res):
    """Calculate the bounding-box of a raster, and return it in a four-element
    tuple: (xmin, ymin, xmax, ymax). This assumes a flat-earth crs to do an
    Affine transformation with the "Augmented Matrix" approach (does not handle
    reprojections):
    https://en.wikipedia.org/wiki/Affine_transformation#Augmented_matrix
    
    Parameters
    ----------
    xs : numpy.array
        1D NumPy array of floats representing the x-values of a raster. This
        likely originated from an xarray.DataArray or xarray.Dataset object
        (xr.open_rasterio).
    ys : numpy.array
        1D NumPy array of floats representing the y-values of a raster. This
        likely originated from an xarray.DataArray or xarray.Dataset object
        (xr.open_rasterio).
    res : tuple
        Two-tuple (int, int) which includes x and y resolutions (aka "grid/cell
        sizes"), respectively.
    """
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    Ab = np.array([[res[0], 0.,      xs.min()],
                   [0.,     -res[1], ys.max()],
                   [0.,     0.,      1.]])
    for x_, y_ in [(0, 0), (0, len(ys)), (len(xs), 0), (len(xs), len(ys))]:
        x, y, _ = np.dot(Ab, np.array([x_, y_, 1.]))
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
    return xmin, ymin, xmax, ymax


def get_indices(x, y, xs, ys, res):
    """Return indices corresponding to transformed coordinates (x, y).
    This assumes that a flat-earth crs to perform an inverse Affine
    transformation, as described here:
    https://en.wikipedia.org/wiki/Affine_transformation#Groups
    It also assumes that the Affine transformation matrix is invertible.
    
    Parameters
    ----------
    x : float
        x-coordinate of the transformed point.
    y : float
        y-coordinate of the transformed point.
    xs : numpy.array
        1D NumPy array of floats representing the x-values of a raster. This
        likely originated from an xarray.DataArray or xarray.Dataset object
        (xr.open_rasterio).
    ys : numpy.array
        1D NumPy array of floats representing the y-values of a raster. This
        likely originated from an xarray.DataArray or xarray.Dataset object
        (xr.open_rasterio).
    res : tuple
        Two-tuple (int, int) which includes x and y resolutions (aka "grid/cell
        sizes"), respectively.
    """
    Ab = np.array([[res[0], 0.,      xs.min()],
                   [0.,     -res[1], ys.max()],
                   [0.,     0.,      1.]])
    nrows, ncols = Ab.shape
    A = Ab[:nrows-1, :ncols-1]
    b = Ab[:, ncols-1][:nrows-1]
    A_inv = np.linalg.inv(A)
    A_ = np.vstack([A_inv, np.zeros(A_inv.shape[1])])
    neg__Ainv__dot__b = -np.dot(A_inv, b)
    b_ = np.hstack([neg__Ainv__dot__b, [1]]).reshape(neg__Ainv__dot__b.shape[0]+1, 1)
    Affine_inv = np.hstack([A_, b_])
    x_, y_, _ = np.dot(Affine_inv, np.array([x, y, 1.]))
    return int(x_), int(y_)


def downsample_aggregate(aggregate, factor, how='mean'):
    """Create downsampled aggregate factor in pixels units"""
    ys, xs = aggregate.shape[:2]
    crarr = aggregate[:ys-(ys % int(factor)), :xs-(xs % int(factor))]
    concat = np.concatenate([[crarr[i::factor, j::factor]
                            for i in range(factor)]
                            for j in range(factor)])

    if how == 'mean':
        return np.nanmean(concat, axis=0)
    elif how == 'sum':
        return np.nansum(concat, axis=0)
    elif how == 'max':
        return np.nanmax(concat, axis=0)
    elif how == 'min':
        return np.nanmin(concat, axis=0)
    elif how == 'median':
        return np.nanmedian(concat, axis=0)
    elif how == 'std':
        return np.nanstd(concat, axis=0)
    elif how == 'var':
        return np.nanvar(concat, axis=0)
    else:
        raise ValueError("Invalid 'how' downsample method. Options mean, sum, max, min, median, std, var")


def summarize_aggregate_values(aggregate, how='linear', num=180):
    """Helper function similar to np.linspace which return values from aggregate min value to aggregate max value in either linear or log space.
    """

    max_val = np.nanmax(aggregate.values)
    min_val = np.nanmin(aggregate.values)

    if min_val == 0:
        min_val = aggregate.data[aggregate.data > 0].min()

    if how == 'linear':
        vals = np.linspace(min_val, max_val, num)[None, :]
    else:
        vals = (np.logspace(0,
                            np.log1p(max_val - min_val),
                            base=np.e, num=num,
                            dtype=min_val.dtype) + min_val)[None, :]

    return DataArray(vals), min_val, max_val


def hold(f):
    '''
    simple arg caching decorator
    '''
    last = []

    def _(*args):
        if not last or last[0] != args:
            last[:] = args, f(*args)
        return last[1]
    return _


def export_image(img, filename, fmt=".png", _return=True, export_path=".", background=""):
    """Given a datashader Image object, saves it to a disk file in the requested format"""

    from datashader.transfer_functions import set_background

    if not os.path.exists(export_path):
        os.mkdir(export_path)

    if background:
        img = set_background(img, background)

    img.to_pil().save(os.path.join(export_path, filename + fmt))
    return img if _return else None


def lnglat_to_meters(longitude, latitude):
    """
    Projects the given (longitude, latitude) values into Web Mercator
    coordinates (meters East of Greenwich and meters North of the Equator).

    Longitude and latitude can be provided as scalars, Pandas columns,
    or Numpy arrays, and will be returned in the same form.  Lists
    or tuples will be converted to Numpy arrays.

    Examples:
       easting, northing = lnglat_to_meters(-40.71,74)

       easting, northing = lnglat_to_meters(np.array([-74]),np.array([40.71]))

       df=pandas.DataFrame(dict(longitude=np.array([-74]),latitude=np.array([40.71])))
       df.loc[:, 'longitude'], df.loc[:, 'latitude'] = lnglat_to_meters(df.longitude,df.latitude)
    """
    if isinstance(longitude, (list, tuple)):
        longitude = np.array(longitude)
    if isinstance(latitude, (list, tuple)):
        latitude = np.array(latitude)

    origin_shift = np.pi * 6378137
    easting = longitude * origin_shift / 180.0
    northing = np.log(np.tan((90 + latitude) * np.pi / 360.0)) * origin_shift / np.pi
    return (easting, northing)
