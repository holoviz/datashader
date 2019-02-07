from __future__ import absolute_import
import copy
import os
import shutil
import json

import numpy as np
import pandas as pd
import dask.dataframe as dd

from datashader.utils import ngjit
from datashader.spatial import hilbert_curve as hc
try:
    import fastparquet as fp
except ImportError:
    fp = None

# Declare Python2/3 unicode-safe string type
try:
    basestring
except NameError:
    basestring = str


def _data2coord(vals, val_range, side_length):
    """
    Convert an array of values from continuous data coordinates to discrete
    Hilbert coordinates

    Parameters
    ----------
    vals: list or tuple or np.ndarray
        Array of continuous data coordinate to be converted to discrete
        distance coordinates
    val_range: tuple
        Start (val_range[0]) and stop (val_range[1]) range in continuous
        data coordinates
    side_length: int
        The number of discrete distance coordinates

    Returns
    -------
    np.ndarray
        Of discrete Hilbert coordinates
    """
    if isinstance(vals, (list, tuple)):
        vals = np.array(vals)

    x_width = val_range[1] - val_range[0]
    return ((vals - val_range[0]) * (side_length / x_width)
            ).astype(np.int64).clip(0, side_length - 1)


def _compute_distance(df, x, y, p, x_range, y_range):
    """
    Compute an array of Hilbert distances from a pandas dataframe

    Parameters
    ----------
    df: pd.DataFrame
        pandas dataframe containing the coordinate columns
    x
        Label of the column containing the x-coordinates
    y
        Label of the column containing the y-coordinates
    p: int
        Hilbert-curve p value
    x_range: tuple
        Start (x_range[0]) and stop (x_range[1]) range of Hilbert x
        scale in data coordinates
    y_range: tuple
        Start (y_range[0]) and stop (y_range[1]) range of Hilbert y
        scale in data coordinates

    Returns
    -------
    np.ndarray
        Of Hilbert distances
    """
    side_length = 2 ** p
    x_coords = _data2coord(df[x], x_range, side_length)
    y_coords = _data2coord(df[y], y_range, side_length)
    return hc.distance_from_coordinates(p, x_coords, y_coords)


def _compute_extents(df, x, y):
    """
    Compute the min and max for x and y columns in a pandas dataframe

    Parameters
    ----------
    df: pd.DataFrame
    x
        Label of column in df that contains the x values
    y
        Label of column in df that contains the y values

    Returns
    -------
    pd.DataFrame
        Single row dataframe containing x_min, x_max, y_min, and y_max columns
    """
    x_min = df[x].min()
    x_max = df[x].max()
    y_min = df[y].min()
    y_max = df[y].max()
    return pd.DataFrame({'x_min': x_min,
                         'x_max': x_max,
                         'y_min': y_min,
                         'y_max': y_max},
                        index=[0])


def _validate_fastparquet():
    """
    Raise an informative error message if fastparquet is not installed
    """
    if fp is None:
        raise ImportError("""\
The datashader.spatial module requires the fastparquet package""")


class SpatialPointsFrame(object):
    """
    Class that wraps a spatially partitioned parquet data set and provides
    a query_partitions method to access the subset of partitions necessary to
    cover the specified x/y range extents.

    The spatially partitioned parquet data set must first be created using
    the `SpatialPointsFrame.partition_and_write` static method. Instances
    of the SpatialPointsFrame class can then be constructed with this
    partitioned parquet data set.

    Note that this class is only suitable for partitioning data sets for use
    with the Canvas.points aggregation method.

    Examples
    --------
    First, construct a spatially partitioned parquet file. This is an expensive
    operation that only needs to be performed once for a data set.
    >>> import numpy as np  # doctest: +SKIP
    ... import pandas as pd
    ... from datashader.spatial import SpatialPointsFrame
    ...
    ... N = 1000
    ... df = pd.DataFrame({
    ...     'x': np.random.rand(N),
    ...     'y': np.random.rand(N) * 2,
    ...     'a': np.random.randn(N)})
    ...
    ... filename = './spatial_points.parquet'
    ... SpatialPointsFrame.partition_and_write(df, 'x', 'y', filename)

    Then, construct a SpatialPointsFrame and use it to extract
    subsets of the original dataframe based on x/y range extents.
    >>> sframe = SpatialPointsFrame(filename, persist=True)  # doctest: +SKIP
    ...
    ... def create_image(x_range, y_range):
    ...     cvs = ds.Canvas(x_range=x_range, y_range=y_range)
    ...     df = sframe.query_partitions(x_range, y_range)
    ...     agg = cvs.points(df, 'x', 'y')
    ...     return tf.dynspread(tf.shade(agg))
    ...
    ... create_image(x_range=(0, 0.3), y_range=(1, 1.5))
    """

    @staticmethod
    def partition_and_write(df,
                            x,
                            y,
                            filename,
                            p=10,
                            npartitions=None,
                            shuffle=None,
                            compression='default'):
        """
        Perform spatial partitioning on an input dataframe and write the
        result to a parquet file.  The resulting parquet file will contain
        the same columns as the input dataframe, but the dataframe's original
        index will be dropped.

        The resulting parquet file will contain all of the rows from the
        input dataframe, but they will be spatially sorted and partitioned
        along a 2D Hilbert curve (https://en.wikipedia.org/wiki/Hilbert_curve).

        The parquet file will also contain custom metadata that is needed to
        reconstruct the Hilbert curve distances on load.  This parquet file
        may then be used to construct SpatialPointsFrame instances.

        Parameters
        ----------
        df: pd.DataFrame or dd.DataFrame
            The input dataframe to partition
        x, y
            The column labels in df of the x and y coordinates of each row
        filename: str
            The path where the resulting parquet file should be written.
            See dask.dataframe.to_parquet for description of supported path
            specifications.
        p: int (default 10)
            The Hilbert curve order parameter that determines the resolution
            of the 2D grid that data points are rounded to before computing
            their Hilbert distance. Points will be discretized into 2 ** p
            bins in each the x and y dimensions.

            This parameter should be increased if the partitions of the
            resulting parquet files are significantly unbalanced.

        npartitions: int or None (default None)
            The number of partitions for the resulting parquet file.  If None
            (the default) this is chosen to be the greater of 8 and
            len(df) // 2**23.

            In general, increasing the number of partitions will improve
            performance when processing small subsets of the overall parquet
            data set.  But this comes at the cost of some additional overhead
            when processing the entire data set.

        shuffle: str or None (default None)
            The dask.dataframe.DataFrame.set_index shuffle method. If None,
            a default is chosen based on the current scheduler.

        compression: str or None (default)
            The dask.dataframe.to_parquet compression method.
        """

        _validate_fastparquet()

        # Validate filename
        if (not isinstance(filename, basestring) or
                not filename.endswith('.parquet')):
            raise ValueError(
                'filename must be a string ending with a .parquet extension')

        # Remove any existing directory
        if os.path.exists(filename):
            shutil.rmtree(filename)

        # Normalize to dask dataframe
        if isinstance(df, pd.DataFrame):
            ddf = dd.from_pandas(df, npartitions=4)
        elif isinstance(df, dd.DataFrame):
            ddf = df
        else:
            raise ValueError("""
df must be a pandas or dask DataFrame instance.
Received value of type {typ}""".format(typ=type(df)))

        # Compute npartitions if needed
        if npartitions is None:
            # Make partitions of ~8 million rows with a minimum of 8
            # partitions
            npartitions = max(len(df) // 2**23, 8)

        # Compute data extents
        extents = ddf.map_partitions(
            _compute_extents, x, y).compute()

        x_range = (float(extents['x_min'].min()),
                   float(extents['x_max'].max()))

        y_range = (float(extents['y_min'].min()),
                   float(extents['y_max'].max()))

        # Compute distance of points along the Hilbert-curve
        ddf = ddf.assign(distance=ddf.map_partitions(
            _compute_distance, x=x, y=y, p=p,
            x_range=x_range, y_range=y_range))

        # Set index to distance. This will trigger an expensive shuffle
        # sort operation
        ddf = ddf.set_index('distance',
                            npartitions=npartitions,
                            shuffle=shuffle)

        # Get list of the distance divisions computed by dask
        distance_divisions = [int(d) for d in ddf.divisions]

        # Save properties as custom metadata in the parquet file
        props = dict(
            version='1.0',
            x=x,
            y=y,
            p=p,
            distance_divisions=distance_divisions,
            x_range=x_range,
            y_range=y_range,
        )

        # Drop distance index to save storage space
        ddf = ddf.reset_index(drop=True)

        # Save ddf to parquet
        dd.to_parquet(
            ddf, filename, engine='fastparquet', compression=compression)

        # Open resulting parquet file
        pf = fp.ParquetFile(filename)

        # Add a new property to the file metadata
        new_fmd = copy.copy(pf.fmd)
        new_kv = fp.parquet_thrift.KeyValue()
        new_kv.key = 'SpatialPointsFrame'
        new_kv.value = json.dumps(props)
        new_fmd.key_value_metadata.append(new_kv)

        # Overwrite file metadata
        fn = os.path.join(filename, '_metadata')
        fp.writer.write_common_metadata(fn, new_fmd, no_row_groups=False)

        fn = os.path.join(filename, '_common_metadata')
        fp.writer.write_common_metadata(fn, new_fmd)

    def __init__(self, filename, persist=False):
        """
        Construct a SpatialPointsFrame from a spatially partitioned parquet
        file

        Parameters
        ----------
        filename: str
            Path to a spatially partitioned parquet file that was created
            using SpatialPointsFrame.partition_and_write
        persist: bool (default False)
            Whether to persist the entire parquet file as a Dask dataframe
            in memory
        """
        _validate_fastparquet()

        # Open parquet file
        pf = fp.ParquetFile(filename)

        # Check for required metadata
        if 'SpatialPointsFrame' not in pf.key_value_metadata:
            raise ValueError("""
The parquet file at '{filename}'
does not appear to be spatially partitioned.
Please construct a spatially partitioned parquet file using the
SpatialPointsFrame.partition_and_write static method.""".format(
                filename=filename))

        # Load metadata
        props = json.loads(pf.key_value_metadata['SpatialPointsFrame'])
        self._x = props['x']
        self._y = props['y']
        self._p = props['p']
        self._x_range = tuple(props['x_range'])
        self._y_range = tuple(props['y_range'])
        self._distance_divisions = tuple(props['distance_divisions'])
        self._npartitions = len(self._distance_divisions) - 1

        # Compute grids
        self._partition_grid = _build_partition_grid(
            self._distance_divisions, self._p)

        # Compute derived properties
        n = 2
        self._side_length = 2 ** self._p
        self._max_distance = 2 ** (n * self._p) - 1
        self._x_width = self._x_range[1] - self._x_range[0]
        self._y_width = self._y_range[1] - self._y_range[0]
        self._x_bin_width = self._x_width / self._side_length
        self._y_bin_width = self._y_width / self._side_length

        # Read parquet file
        self._frame = dd.read_parquet(filename)

        # Persist if requested
        if persist:
            self._frame = self._frame.persist()

    def query_partitions(self, x_range, y_range):
        """
        Query the underlying parquet file for the partitions that potentially
        intersect with a given rectangular region.

        Parameters
        ----------
        x_range, y_range: tuple
            Length-2 tuples containing the x and y extents of the query region

        Returns
        -------
        dd.DataFrame
            Dask dataframe containing all data from all partitions that
            potentially intersect with the specified x_range/y_range box.
        """
        # Expand upper range to account for rounding
        expanded_x_range = [x_range[0],
                            x_range[1] + self._x_bin_width]

        expanded_y_range = [y_range[0],
                            y_range[1] + self._y_bin_width]

        # Compute ranges in integer coordinates
        query_x_range_coord = _data2coord(expanded_x_range,
                                          self._x_range,
                                          self._side_length)

        query_y_range_coord = _data2coord(expanded_y_range,
                                          self._y_range,
                                          self._side_length)

        # Get corresponding slice of partition grid
        partition_query = self._partition_grid[
            slice(*query_x_range_coord), slice(*query_y_range_coord)]

        # Get unique partitions present in slice
        query_partitions = sorted(np.unique(partition_query))

        if query_partitions:
            partition_dfs = [self._frame.get_partition(p)
                             for p in query_partitions]
            query_frame = dd.concat(partition_dfs)
            return query_frame
        else:
            # return an empty Dask dataframe with the right shape
            return (self._frame
                    .get_partition(0)
                    .map_partitions(lambda df: df.iloc[1:0]))

    # Read-only properties
    @property
    def frame(self):
        """
        Dask dataframe backed by a spatially partitioned parquet file

        Returns
        -------
        dd.DataFrame
        """
        return self._frame

    @property
    def hilbert_distances(self):
        """
        Dask series containing the Hilbert distance of each row in frame

        Returns
        -------
        dd.Series
        """
        x = self.x
        y = self._y
        p = self._p
        x_range = self._x_range
        y_range = self._y_range
        return self._frame.map_partitions(
            _compute_distance, x=x, y=y, p=p,
            x_range=x_range, y_range=y_range)

    @property
    def partitions(self):
        """
        Dask series containing the partition of each row in frame

        Returns
        -------
        dd.Series
        """
        search_divisions = np.array(self.distance_divisions[:-1])
        return self.hilbert_distances.map_partitions(
            lambda s: pd.Series(
                np.ones(len(s)) *
                search_divisions.searchsorted(s.iloc[0], side='right'),
                dtype='int64',
                index=s.index))

    @property
    def x(self):
        """
        Column label in frame containing the x coordinate of each row
        """
        return self._x

    @property
    def y(self):
        """
        Column label in frame containing the x coordinate of each row
        """
        return self._y

    @property
    def p(self):
        """
        Hilbert curve order parameter

        Returns
        -------
        int
        """
        return self._p

    @property
    def x_range(self):
        """
        x range extents for the entire dataset

        Returns
        -------
        x_range: tuple
            The min (x_range[0]) and max (x_range[1]) x coordinates in frame
        """
        return self._x_range

    @property
    def y_range(self):
        """
        y range extents for the entire dataset

        Returns
        -------
        y_range: tuple
            The min (y_range[0]) and max (y_range[1]) y coordinates in frame
        """
        return self._y_range

    @property
    def x_bin_edges(self):
        """
        Array of the discrete x-coordinates that points are rounded to
        in order to compute their Hilbert curve distance

        Returns
        -------
        x_bin_edges: np.ndarray
        """
        return np.linspace(*self.x_range, self._side_length + 1)

    @property
    def y_bin_edges(self):
        """
        Array of the discrete y-coordinates that points are rounded to
        in order to compute their Hilbert curve distance

        Returns
        -------
        y_bin_edges: np.ndarray
        """
        return np.linspace(*self.y_range, self._side_length + 1)

    @property
    def distance_divisions(self):
        """
        tuple of the Hilbert distance divisions corresponding to the
        partitions in frame

        Returns
        -------
        tuple
        """
        return self._distance_divisions

    @property
    def npartitions(self):
        """
        The number of partitions in frame

        Returns
        -------
        int
        """
        return self._npartitions


@ngjit
def _build_distance_grid(p):
    """
    Build a (2 ** p) x (2 ** p) array containing the Hilbert distance of
    each discrete location in the grid

    Parameters
    ----------
    p: int
        Hilbert curve order

    Returns
    -------
    np.ndarray
        2D array containing the Hilbert curve distances for a curve with
        order p
    """
    side_length = int(2 ** p)
    distance_grid = np.zeros((side_length, side_length), dtype=np.int64)
    for i in range(side_length):
        for j in range(side_length):
            distance_grid[i, j] = (
                hc.distance_from_coordinates(p, i, j))
    return distance_grid


@ngjit
def _build_partition_grid(dask_divisions, p):
    """
    Build a (2 ** p) x (2 ** p) array containing the partition number
    of each discrete location in the grid

    Parameters
    ----------
    dask_divisions: list of int
        Hilbert distance divisions for the parquet file
    p: int
        Hilbert curve order

    Returns
    -------
    np.ndarray
        2D array containing the Hilbert distance partitions for a curve
        with order p
    """
    distance_grid = _build_distance_grid(p)
    search_divisions = np.array(
        list(dask_divisions[1:-1]))

    side_length = 2 ** p
    partition_grid = np.zeros((side_length, side_length), dtype=np.int64)
    for i in range(side_length):
        for j in range(side_length):
            partition_grid[i, j] = np.searchsorted(
                search_divisions,
                distance_grid[i, j],
                side='right')
    return partition_grid
