from __future__ import annotations

import os
import re

from inspect import getmro

import numba as nb
import numpy as np
import pandas as pd

from toolz import memoize
from xarray import DataArray

import dask.dataframe as dd
import datashape

try:
    from datashader.datatypes import RaggedDtype
except ImportError:
    RaggedDtype = type(None)

try:
    import cudf
except Exception:
    cudf = None

try:
    from spatialpandas.geometry import GeometryDtype
except ImportError:
    GeometryDtype = type(None)


class VisibleDeprecationWarning(UserWarning):
    """Visible deprecation warning.

    By default, python will not show deprecation warnings, so this class
    can be used when a very visible warning is helpful, for example because
    the usage is most likely a user bug.
    """


ngjit = nb.jit(nopython=True, nogil=True)
ngjit_parallel = nb.jit(nopython=True, nogil=True, parallel=True)

# Get and save the Numba version, will be used to limit functionality
numba_version = tuple([int(x) for x in re.match(
                            r"([0-9]+)\.([0-9]+)\.([0-9]+)",
                            nb.__version__).groups()])


class Expr(object):
    """Base class for expression-like objects.

    Implements hashing and equality checks. Subclasses should implement an
    ``inputs`` attribute/property, containing a tuple of everything that fully
    defines that expression.
    """
    def __hash__(self):
        return hash((type(self), self._hashable_inputs()))

    def __eq__(self, other):
        return (type(self) is type(other) and
                self._hashable_inputs() == other._hashable_inputs())

    def __ne__(self, other):
        return not self == other

    def _hashable_inputs(self):
        """
        Return a version of the inputs tuple that is suitable for hashing and
        equality comparisons
        """
        result = []
        for ip in self.inputs:
            if isinstance(ip, (list, set)):
                result.append(tuple(ip))
            elif isinstance(ip, np.ndarray):
                result.append(ip.tobytes())
            else:
                result.append(ip)

        return tuple(result)


class Dispatcher(object):
    """Simple single dispatch."""
    def __init__(self):
        self._lookup = {}

    def register(self, typ, func=None):
        """Register dispatch of `func` on arguments of type `typ`"""
        if func is None:
            return lambda f: self.register(typ, f)
        if isinstance(typ, tuple):
            for t in typ:
                self.register(t, func)
        else:
            self._lookup[typ] = func
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


def isrealfloat(dt):
    """Check if a datashape is numeric and real.

    Example
    -------
    >>> isrealfloat('int32')
    False
    >>> isrealfloat('float64')
    True
    >>> isrealfloat('string')
    False
    >>> isrealfloat('complex64')
    False
    """
    dt = datashape.predicates.launder(dt)
    return isinstance(dt, datashape.Unit) and dt in datashape.typesets.floating


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
    dt = datashape.predicates.launder(dt)
    return isinstance(dt, datashape.Unit) and dt in datashape.typesets.real


def nansum_missing(array, axis):
    """nansum where all-NaN values remain NaNs.

    Note: In NumPy <=1.9 NaN is returned for slices that are
    all NaN, while later versions return 0. This function emulates
    the older behavior, which allows using NaN as a missing value
    indicator.

    Parameters
    ----------
    array: Array to sum over
    axis:  Axis to sum over
    """
    T = list(range(array.ndim))
    T.remove(axis)
    T.insert(0, axis)
    array = array.transpose(T)
    missing_vals = np.isnan(array)
    all_empty = np.all(missing_vals, axis=0)
    set_to_zero = missing_vals & ~all_empty
    return np.where(set_to_zero, 0, array).sum(axis=0)


def calc_res(raster):
    """Calculate the resolution of xarray.DataArray raster and return it as the
    two-tuple (xres, yres). yres is positive if it is decreasing.
    """
    h, w = raster.shape[-2:]
    ydim, xdim = raster.dims[-2:]
    xcoords = raster[xdim].values
    ycoords = raster[ydim].values
    xres = (xcoords[-1] - xcoords[0]) / (w - 1)
    yres = (ycoords[0] - ycoords[-1]) / (h - 1)
    return xres, yres


def calc_bbox(xs, ys, res):
    """Calculate the bounding box of a raster, and return it in a four-element
    tuple: (xmin, ymin, xmax, ymax). This calculation assumes the raster is
    uniformly sampled (equivalent to a flat-earth assumption, for geographic
    data) so that an affine transform (using the "Augmented Matrix" approach)
    suffices:
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
    xbound = xs.max() if res[0] < 0 else xs.min()
    ybound = ys.min() if res[1] < 0 else ys.max()

    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    Ab = np.array([[res[0],  0.,      xbound],
                   [0.,      -res[1], ybound],
                   [0.,      0.,      1.]])
    for x_, y_ in [(0, 0), (0, len(ys)), (len(xs), 0), (len(xs), len(ys))]:
        x, y, _ = np.dot(Ab, np.array([x_, y_, 1.]))
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
    xpad, ypad = res[0]/2., res[1]/2.
    return xmin-xpad, ymin+ypad, xmax-xpad, ymax+ypad


def get_indices(start, end, coords, res):
    """
    Transform continuous start and end coordinates into array indices.

    Parameters
    ----------
    start : float
        coordinate of the lower bound.
    end : float
        coordinate of the upper bound.
    coords : numpy.ndarray
        coordinate values along the axis.
    res : tuple
        Resolution along an axis (aka "grid/cell sizes")
    """
    size = len(coords)
    half = abs(res)/2.
    vmin, vmax = coords.min(), coords.max()
    span = vmax-vmin
    start, end = start+half-vmin, end-half-vmin
    sidx, eidx = int((start/span)*size), int((end/span)*size)
    if eidx < sidx:
        return sidx, sidx
    return sidx, eidx


def _flip_array(array, xflip, yflip):
    # array may have 2 or 3 dimensions, last one is x-dimension, last but one is y-dimension.
    if yflip:
        array = array[..., ::-1, :]
    if xflip:
        array = array[..., :, ::-1]
    return array


def orient_array(raster, res=None, layer=None):
    """
    Reorients the array to a canonical orientation depending on
    whether the x and y-resolution values are positive or negative.

    Parameters
    ----------
    raster : DataArray
        xarray DataArray to be reoriented
    res : tuple
        Two-tuple (int, int) which includes x and y resolutions (aka "grid/cell
        sizes"), respectively.
    layer : int
        Index of the raster layer to be reoriented (optional)

    Returns
    -------
    array : numpy.ndarray
        Reoriented 2d NumPy ndarray
    """
    if res is None:
        res = calc_res(raster)
    array = raster.data
    if layer is not None: array = array[layer-1]
    r0zero = np.timedelta64(0, 'ns') if isinstance(res[0], np.timedelta64) else 0
    r1zero = np.timedelta64(0, 'ns') if isinstance(res[1], np.timedelta64) else 0
    xflip = res[0] < r0zero
    yflip = res[1] > r1zero
    array = _flip_array(array, xflip, yflip)
    return array


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
       easting, northing = lnglat_to_meters(-74,40.71)

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


# Heavily inspired by odo
def dshape_from_pandas_helper(col):
    """Return an object from datashape.coretypes given a column from a pandas
    dataframe.
    """
    if (isinstance(col.dtype, type(pd.Categorical.dtype)) or
            isinstance(col.dtype, pd.api.types.CategoricalDtype) or
            cudf and isinstance(col.dtype, cudf.core.dtypes.CategoricalDtype)):
        # Compute category dtype
        pd_categories = col.cat.categories
        if isinstance(pd_categories, dd.Index):
            pd_categories = pd_categories.compute()
        if cudf and isinstance(pd_categories, cudf.Index):
            pd_categories = pd_categories.to_pandas()

        categories = np.array(pd_categories)

        if categories.dtype.kind == 'U':
            categories = categories.astype('object')

        cat_dshape = datashape.dshape('{} * {}'.format(
            len(col.cat.categories),
            categories.dtype,
        ))
        return datashape.Categorical(categories,
                                     type=cat_dshape,
                                     ordered=col.cat.ordered)
    elif col.dtype.kind == 'M':
        tz = getattr(col.dtype, 'tz', None)
        if tz is not None:
            # Pandas stores this as a pytz.tzinfo, but DataShape wants a string
            tz = str(tz)
        return datashape.Option(datashape.DateTime(tz=tz))
    elif isinstance(col.dtype, (RaggedDtype, GeometryDtype)):
        return col.dtype
    dshape = datashape.CType.from_numpy_dtype(col.dtype)
    dshape = datashape.string if dshape == datashape.object_ else dshape
    if dshape in (datashape.string, datashape.datetime_):
        return datashape.Option(dshape)
    return dshape


def dshape_from_pandas(df):
    """Return a datashape.DataShape object given a pandas dataframe."""
    return len(df) * datashape.Record([(k, dshape_from_pandas_helper(df[k]))
                                       for k in df.columns])


@memoize(key=lambda args, kwargs: tuple(args[0].__dask_keys__()))
def dshape_from_dask(df):
    """Return a datashape.DataShape object given a dask dataframe."""
    cat_columns = [
        col for col in df.columns
        if (isinstance(df[col].dtype, type(pd.Categorical.dtype)) or
            isinstance(df[col].dtype, pd.api.types.CategoricalDtype))
           and not getattr(df[col].cat, 'known', True)]
    df = df.categorize(cat_columns, index=False)
    # get_partition(0) used below because categories are sometimes repeated
    # for dask-cudf DataFrames with multiple partitions
    return datashape.var * datashape.Record([
        (k, dshape_from_pandas_helper(df[k].get_partition(0))) for k in df.columns
    ]), df


def dshape_from_xarray_dataset(xr_ds):
    """Return a datashape.DataShape object given a xarray Dataset."""
    return datashape.var * datashape.Record([
        (k, dshape_from_pandas_helper(xr_ds[k]))
        for k in list(xr_ds.data_vars) + list(xr_ds.coords)
    ])


def dataframe_from_multiple_sequences(x_values, y_values):
   """
   Converts a set of multiple sequences (eg: time series), stored as a 2 dimensional
   numpy array into a pandas dataframe that can be plotted by datashader.
   The pandas dataframe eventually contains two columns ('x' and 'y') with the data.
   Each time series is separated by a row of NaNs.
   Discussion at: https://github.com/bokeh/datashader/issues/286#issuecomment-334619499

   x_values: 1D numpy array with the values to be plotted on the x axis (eg: time)
   y_values: 2D numpy array with the sequences to be plotted of shape (num sequences X length of each sequence)

   """

   # Add a NaN at the end of the array of x values
   x = np.zeros(x_values.shape[0] + 1)
   x[-1] = np.nan
   x[:-1] = x_values

   # Tile this array of x values: number of repeats = number of sequences/time series in the data
   x = np.tile(x, y_values.shape[0])

   # Add a NaN at the end of every sequence in y_values
   y = np.zeros((y_values.shape[0], y_values.shape[1] + 1))
   y[:, -1] = np.nan
   y[:, :-1] = y_values

   # Return a dataframe with this new set of x and y values
   return pd.DataFrame({'x': x, 'y': y.flatten()})


def _pd_mesh(vertices, simplices):
    """Helper for ``datashader.utils.mesh()``. Both arguments are assumed to be
    Pandas DataFrame objects.
    """
    # Winding auto-detect
    winding = [0, 1, 2]
    first_tri = vertices.values[simplices.values[0, winding].astype(np.int64), :2]
    a, b, c = first_tri
    if np.cross(b-a, c-a).item() >= 0:
        winding = [0, 2, 1]

    # Construct mesh by indexing into vertices with simplex indices
    vertex_idxs = simplices.values[:, winding]
    if not vertex_idxs.dtype == 'int64':
        vertex_idxs = vertex_idxs.astype(np.int64)
    vals = np.take(vertices.values, vertex_idxs, axis=0)
    vals = vals.reshape(np.prod(vals.shape[:2]), vals.shape[2])
    res = pd.DataFrame(vals, columns=vertices.columns)

    # If vertices don't have weights, use simplex weights
    verts_have_weights = len(vertices.columns) > 2
    if not verts_have_weights:
        weight_col = simplices.columns[3]
        res[weight_col] = simplices.values[:, 3].repeat(3)

    return res


def _dd_mesh(vertices, simplices):
    """Helper for ``datashader.utils.mesh()``. Both arguments are assumed to be
    Dask DataFrame objects.
    """
    # Construct mesh by indexing into vertices with simplex indices
    # TODO: For dask: avoid .compute() calls
    res = _pd_mesh(vertices.compute(), simplices.compute())

    # Compute a chunksize that will not split the vertices of a single
    # triangle across partitions
    approx_npartitions = max(vertices.npartitions, simplices.npartitions)
    chunksize = int(np.ceil(len(res) / (3*approx_npartitions)) * 3)

    # Create dask dataframe
    res = dd.from_pandas(res, chunksize=chunksize)
    return res


def mesh(vertices, simplices):
    """Merge vertices and simplices into a triangular mesh, suitable to be
    passed into the ``Canvas.trimesh()`` method via the ``mesh``
    keyword-argument. Both arguments are assumed to be Dask DataFrame
    objects.
    """
    # Verify the simplex data structure
    assert simplices.values.shape[1] >= 3, ('At least three vertex columns '
                                            'are required for the triangle '
                                            'definition')
    simplices_all_ints = simplices.dtypes.iloc[:3].map(
        lambda dt: np.issubdtype(dt, np.integer)
    ).all()
    assert simplices_all_ints, ('Simplices must be integral. You may '
                                'consider casting simplices to integers '
                                'with ".astype(int)"')

    assert len(vertices.columns) > 2 or simplices.values.shape[1] > 3, 'If no vertex weight column is provided, a triangle weight column is required.'


    if isinstance(vertices, dd.DataFrame) and isinstance(simplices, dd.DataFrame):
        return _dd_mesh(vertices, simplices)

    return _pd_mesh(vertices, simplices)


def apply(func, args, kwargs=None):
    if kwargs:
        return func(*args, **kwargs)
    else:
        return func(*args)


@ngjit
def isnull(val):
    """
    Equivalent to isnan for floats, but also numba compatible with integers
    """
    return not (val <= 0 or val > 0)


@ngjit
def isminus1(val):
    """
    Check for -1 which is equivalent to NaN for some integer aggregations
    """
    return val == -1


@ngjit_parallel
def nanfirst_in_place(ret, other):
    """First of 2 arrays but taking nans into account.
    Return the first array.
    """
    ret = ret.ravel()
    other = other.ravel()
    for i in nb.prange(len(ret)):
        if isnull(ret[i]) and not isnull(other[i]):
            ret[i] = other[i]


@ngjit_parallel
def nanlast_in_place(ret, other):
    """Last of 2 arrays but taking nans into account.
    Return the first array.
    """
    ret = ret.ravel()
    other = other.ravel()
    for i in nb.prange(len(ret)):
        if not isnull(other[i]):
            ret[i] = other[i]


@ngjit_parallel
def nanmax_in_place(ret, other):
    """Max of 2 arrays but taking nans into account.  Could use np.nanmax but
    would need to replace zeros with nans where both arrays are nans.
    Return the first array.
    """
    ret = ret.ravel()
    other = other.ravel()
    for i in nb.prange(len(ret)):
        if isnull(ret[i]):
            if not isnull(other[i]):
                ret[i] = other[i]
        elif not isnull(other[i]) and other[i] > ret[i]:
            ret[i] = other[i]


@ngjit_parallel
def nanmin_in_place(ret, other):
    """Min of 2 arrays but taking nans into account.  Could use np.nanmin but
    would need to replace zeros with nans where both arrays are nans.
    Accepts 3D (ny, nx, ncat) and 2D (ny, nx) arrays.
    Return the first array.
    """
    ret = ret.ravel()
    other = other.ravel()
    for i in nb.prange(len(ret)):
        if isnull(ret[i]):
            if not isnull(other[i]):
                ret[i] = other[i]
        elif not isnull(other[i]) and other[i] < ret[i]:
            ret[i] = other[i]


@ngjit
def shift_and_insert(target, value, index):
    """Insert a value into a 1D array at a particular index, but before doing
    that shift the previous values along one to make room. For use in
    ``FloatingNReduction`` classes such as ``max_n`` and ``first_n`` which
    store ``n`` values per pixel.

    Parameters
    ----------
    target : 1d numpy array
        Target pixel array.

    value : float
        Value to insert into target pixel array.

    index : int
        Index to insert at.

    Returns
    -------
    Index beyond insertion, i.e. where the first shifted value now sits.
    """
    n = len(target)
    for i in range(n-1, index, -1):
        target[i] = target[i-1]
    target[index] = value
    return index + 1


@ngjit
def _nanfirst_n_impl(ret_pixel, other_pixel):
    """Single pixel implementation of nanfirst_n_in_place.
    ret_pixel and other_pixel are both 1D arrays of the same length.

    Walk along other_pixel a value at a time, find insertion index in
    ret_pixel and shift values along to insert.  Next other_pixel value is
    inserted at a higher index, so this walks the two pixel arrays just once
    each.
    """
    n = len(ret_pixel)
    istart = 0
    for other_value in other_pixel:
        if isnull(other_value):
            break
        else:
            for i in range(istart, n):
                if isnull(ret_pixel[i]):
                    # Always insert after existing values, so no shifting required.
                    ret_pixel[i] = other_value
                    istart = i+1
                    break


@ngjit_parallel
def nanfirst_n_in_place_4d(ret, other):
    """3d version of nanfirst_n_in_place_4d, taking arrays of shape (ny, nx, n).
    """
    ny, nx, ncat, _n = ret.shape
    for y in nb.prange(ny):
        for x in range(nx):
            for cat in range(ncat):
                _nanfirst_n_impl(ret[y, x, cat], other[y, x, cat])


@ngjit_parallel
def nanfirst_n_in_place_3d(ret, other):
    """3d version of nanfirst_n_in_place_4d, taking arrays of shape (ny, nx, n).
    """
    ny, nx, _n = ret.shape
    for y in nb.prange(ny):
        for x in range(nx):
            _nanfirst_n_impl(ret[y, x], other[y, x])


@ngjit
def _nanlast_n_impl(ret_pixel, other_pixel):
    """Single pixel implementation of nanlast_n_in_place.
    ret_pixel and other_pixel are both 1D arrays of the same length.

    Walk along other_pixel a value at a time, find insertion index in
    ret_pixel and shift values along to insert.  Next other_pixel value is
    inserted at a higher index, so this walks the two pixel arrays just once
    each.
    """
    n = len(ret_pixel)
    istart = 0
    for other_value in other_pixel:
        if isnull(other_value):
            break
        else:
            for i in range(istart, n):
                # Always insert at istart index.
                istart = shift_and_insert(ret_pixel, other_value, istart)
                break


@ngjit_parallel
def nanlast_n_in_place_4d(ret, other):
    """3d version of nanfirst_n_in_place_4d, taking arrays of shape (ny, nx, n).
    """
    ny, nx, ncat, _n = ret.shape
    for y in nb.prange(ny):
        for x in range(nx):
            for cat in range(ncat):
                _nanlast_n_impl(ret[y, x, cat], other[y, x, cat])


@ngjit_parallel
def nanlast_n_in_place_3d(ret, other):
    """3d version of nanlast_n_in_place_4d, taking arrays of shape (ny, nx, n).
    """
    ny, nx, _n = ret.shape
    for y in nb.prange(ny):
        for x in range(nx):
            _nanlast_n_impl(ret[y, x], other[y, x])


@ngjit
def _nanmax_n_impl(ret_pixel, other_pixel):
    """Single pixel implementation of nanmax_n_in_place.
    ret_pixel and other_pixel are both 1D arrays of the same length.

    Walk along other_pixel a value at a time, find insertion index in
    ret_pixel and shift values along to insert.  Next other_pixel value is
    inserted at a higher index, so this walks the two pixel arrays just once
    each.
    """
    n = len(ret_pixel)
    istart = 0
    for other_value in other_pixel:
        if isnull(other_value):
            break
        else:
            for i in range(istart, n):
                if isnull(ret_pixel[i]) or other_value > ret_pixel[i]:
                    istart = shift_and_insert(ret_pixel, other_value, i)
                    break


@ngjit_parallel
def nanmax_n_in_place_4d(ret, other):
    """Combine two max-n arrays, taking nans into account. Max-n arrays are 4D
    with shape (ny, nx, ncat, n) where ny and nx are the number of pixels,
    ncat the number of categories (will be 1 if not using a categorical
    reduction) and the last axis containing n values in descending order.
    If there are fewer than n values it is padded with nans.
    Return the first array.
    """
    ny, nx, ncat, _n = ret.shape
    for y in nb.prange(ny):
        for x in range(nx):
            for cat in range(ncat):
                _nanmax_n_impl(ret[y, x, cat], other[y, x, cat])


@ngjit_parallel
def nanmax_n_in_place_3d(ret, other):
    """3d version of nanmax_n_in_place_4d, taking arrays of shape (ny, nx, n).
    """
    ny, nx, _n = ret.shape
    for y in nb.prange(ny):
        for x in range(nx):
            _nanmax_n_impl(ret[y, x], other[y, x])


@ngjit
def _nanmin_n_impl(ret_pixel, other_pixel):
    """Single pixel implementation of nanmin_n_in_place.
    ret_pixel and other_pixel are both 1D arrays of the same length.

    Walk along other_pixel a value at a time, find insertion index in
    ret_pixel and shift values along to insert.  Next other_pixel value is
    inserted at a higher index, so this walks the two pixel arrays just once
    each.
    """
    n = len(ret_pixel)
    istart = 0
    for other_value in other_pixel:
        if isnull(other_value):
            break
        else:
            for i in range(istart, n):
                if isnull(ret_pixel[i]) or other_value < ret_pixel[i]:
                    istart = shift_and_insert(ret_pixel, other_value, i)
                    break


@ngjit_parallel
def nanmin_n_in_place_4d(ret, other):
    """Combine two min-n arrays, taking nans into account. Min-n arrays are 4D
    with shape (ny, nx, ncat, n) where ny and nx are the number of pixels,
    ncat the number of categories (will be 1 if not using a categorical
    reduction) and the last axis containing n values in ascending order.
    If there are fewer than n values it is padded with nans.
    Return the first array.
    """
    ny, nx, ncat, _n = ret.shape
    for y in nb.prange(ny):
        for x in range(nx):
            for cat in range(ncat):
                _nanmin_n_impl(ret[y, x, cat], other[y, x, cat])


@ngjit_parallel
def nanmin_n_in_place_3d(ret, other):
    """3d version of nanmin_n_in_place_4d, taking arrays of shape (ny, nx, n).
    """
    ny, nx, _n = ret.shape
    for y in nb.prange(ny):
        for x in range(nx):
            _nanmin_n_impl(ret[y, x], other[y, x])


@ngjit_parallel
def nansum_in_place(ret, other):
    """Sum of 2 arrays but taking nans into account.  Could use np.nansum but
    would need to replace zeros with nans where both arrays are nans.
    Return the first array.
    """
    ret = ret.ravel()
    other = other.ravel()
    for i in nb.prange(len(ret)):
        if isnull(ret[i]):
            if not isnull(other[i]):
                ret[i] = other[i]
        elif not isnull(other[i]):
            ret[i] += other[i]


@ngjit
def row_max_in_place(ret, other):
    """Maximum of 2 arrays of row indexes.
    Row indexes are integers from 0 upwards, missing data is -1.
    Return the first array.
    """
    ret = ret.ravel()
    other = other.ravel()
    for i in range(len(ret)):
        if other[i] > -1 and (ret[i] == -1 or other[i] > ret[i]):
            ret[i] = other[i]


@ngjit
def row_min_in_place(ret, other):
    """Minimum of 2 arrays of row indexes.
    Row indexes are integers from 0 upwards, missing data is -1.
    Return the first array.
    """
    ret = ret.ravel()
    other = other.ravel()
    for i in range(len(ret)):
        if other[i] > -1 and (ret[i] == -1 or other[i] < ret[i]):
            ret[i] = other[i]


@ngjit
def _row_max_n_impl(ret_pixel, other_pixel):
    """Single pixel implementation of row_max_n_in_place.
    ret_pixel and other_pixel are both 1D arrays of the same length.

    Walk along other_pixel a value at a time, find insertion index in
    ret_pixel and shift values along to insert.  Next other_pixel value is
    inserted at a higher index, so this walks the two pixel arrays just once
    each.
    """
    n = len(ret_pixel)
    istart = 0
    for other_value in other_pixel:
        if other_value == -1:
            break
        else:
            for i in range(istart, n):
                if ret_pixel[i] == -1 or other_value > ret_pixel[i]:
                    istart = shift_and_insert(ret_pixel, other_value, i)
                    break


@ngjit
def row_max_n_in_place_4d(ret, other):
    """Combine two row_max_n signed integer arrays.
    Equivalent to nanmax_n_in_place with -1 replacing NaN for missing data.
    Return the first array.
    """
    ny, nx, ncat, _n = ret.shape
    for y in range(ny):
        for x in range(nx):
            for cat in range(ncat):
                _row_max_n_impl(ret[y, x, cat], other[y, x, cat])


@ngjit
def row_max_n_in_place_3d(ret, other):
    ny, nx, _n = ret.shape
    for y in range(ny):
        for x in range(nx):
            _row_max_n_impl(ret[y, x], other[y, x])


@ngjit
def _row_min_n_impl(ret_pixel, other_pixel):
    """Single pixel implementation of row_min_n_in_place.
    ret_pixel and other_pixel are both 1D arrays of the same length.

    Walk along other_pixel a value at a time, find insertion index in
    ret_pixel and shift values along to insert.  Next other_pixel value is
    inserted at a higher index, so this walks the two pixel arrays just once
    each.
    """
    n = len(ret_pixel)
    istart = 0
    for other_value in other_pixel:
        if other_value == -1:
            break
        else:
            for i in range(istart, n):
                if ret_pixel[i] == -1 or other_value < ret_pixel[i]:
                    istart = shift_and_insert(ret_pixel, other_value, i)
                    break


@ngjit
def row_min_n_in_place_4d(ret, other):
    """Combine two row_min_n signed integer arrays.
    Equivalent to nanmin_n_in_place with -1 replacing NaN for missing data.
    Return the first array.
    """
    ny, nx, ncat, _n = ret.shape
    for y in range(ny):
        for x in range(nx):
            for cat in range(ncat):
                _row_min_n_impl(ret[y, x, cat], other[y, x, cat])


@ngjit
def row_min_n_in_place_3d(ret, other):
    ny, nx, _n = ret.shape
    for y in range(ny):
        for x in range(nx):
            _row_min_n_impl(ret[y, x], other[y, x])
