from __future__ import absolute_import
import os
import pytest
import numpy as np
import pandas as pd
import dask.dataframe as dd

from datashader import Canvas
import datashader.spatial.points as dsp

pytest.importorskip('fastparquet')


@pytest.fixture()
def df():
    N = 1000
    np.random.seed(25)

    df = pd.DataFrame({
        'x': np.random.rand(N),
        'y': np.random.rand(N) * 2,
        'a': np.random.randn(N)
    })

    # Make sure we have x/y values of 0 and 1 represented so that
    # autocomputed ranges are predictable
    df.x.iloc[0] = 0.0
    df.x.iloc[-1] = 1.0
    df.y.iloc[0] = 0.0
    df.y.iloc[-1] = 2.0
    return df


@pytest.fixture(params=[False, True])
def s_points_frame(request, tmp_path, df):

    # Work around https://bugs.python.org/issue33617
    tmp_path = str(tmp_path)
    p = 5
    path = os.path.join(tmp_path, 'spatial_points.parquet')

    dsp.to_parquet(
        df, path, 'x', 'y', p=p, npartitions=10)

    spf = dsp.read_parquet(path)

    if request.param:
        spf = spf.persist()

    return spf


def test_spatial_points_frame_properties(s_points_frame):
    assert s_points_frame.spatial.x == 'x'
    assert s_points_frame.spatial.y == 'y'
    assert s_points_frame.spatial.p == 5
    assert s_points_frame.npartitions == 10
    assert s_points_frame.spatial.x_range == (0, 1)
    assert s_points_frame.spatial.y_range == (0, 2)
    assert s_points_frame.spatial.nrows == 1000

    # x_bin_edges
    np.testing.assert_array_equal(
        s_points_frame.spatial.x_bin_edges,
        np.linspace(0.0, 1.0, 2 ** 5 + 1))

    # y_bin_edges
    np.testing.assert_array_equal(
        s_points_frame.spatial.y_bin_edges,
        np.linspace(0.0, 2.0, 2 ** 5 + 1))

    # distance_divisions
    distance_divisions = s_points_frame.spatial.distance_divisions
    assert len(distance_divisions) == 10 + 1


@pytest.mark.parametrize('x_range,y_range', [
    ((0, 0.2), (0, 0.2)),
    ((0.3, 1.0), (0.5, 1.5)),
    ((5, 10), (5, 10))  # Outside of bounds
])
def test_query_partitions(s_points_frame, x_range, y_range):

    # Get original
    ddf = s_points_frame

    # Query subset
    query_ddf = s_points_frame.spatial_query(x_range, y_range)

    # Make sure we have less partitions
    assert query_ddf.npartitions < ddf.npartitions

    # Make sure query has less rows
    assert len(query_ddf) < len(ddf)

    # Make sure query includes all of the rows in the original that
    # reside in the query region
    df = ddf.compute()
    query_df = query_ddf.compute()

    range_inds = (
        (x_range[0] <= df.x) & (df.x <= x_range[1]) &
        (y_range[0] <= df.y) & (df.y <= y_range[1]))

    query_range_inds = (
        (x_range[0] <= query_df.x) & (query_df.x <= x_range[1]) &
        (y_range[0] <= query_df.y) & (query_df.y <= y_range[1]))

    # Check that the two methods produce the same set of rows
    df1 = df.loc[range_inds].sort_values(['x', 'y', 'a'])
    df2 = query_df.loc[query_range_inds].sort_values(['x', 'y', 'a'])
    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.parametrize('x_range,y_range', [
    ((0, 0.2), (0, 0.2)),
    ((0.3, 1.0), (0.5, 1.5)),
    ((5, 10), (5, 10))  # Outside of bounds
])
def test_aggregation_partitions(s_points_frame, x_range, y_range):
    # Get original as pandas
    df = s_points_frame.compute()

    # Query subset
    query_ddf = s_points_frame.spatial_query(x_range, y_range)

    # Create canvas
    cvs = Canvas(x_range=x_range, y_range=y_range)

    # Aggregate with full pandas frame
    agg_expected = cvs.points(df, 'x', 'y')
    agg_query = cvs.points(query_ddf, 'x', 'y')
    agg = cvs.points(s_points_frame, 'x', 'y')

    assert agg.equals(agg_expected)
    assert agg.equals(agg_query)


def test_validate_parquet_file(df, tmp_path):
    # Work around https://bugs.python.org/issue33617
    tmp_path = str(tmp_path)

    # Write DataFrame to parquet
    filename = os.path.join(tmp_path, 'df.parquet')
    ddf = dd.from_pandas(df, npartitions=4)
    ddf.to_parquet(filename, engine='fastparquet')

    # Try to construct a SpatialPointsFrame from it
    spf = dsp.read_parquet(filename)

    assert spf.spatial is None


def test_filesystem_protocol(df, tmp_path):
    # For now, hardcodes "tmp_path" to force the path to be POSIX; non-POSIX paths (from a real tmp_path) not yet supported.
    p = 5
    # Use an in-memory filesystem protocol
    path = "memory://tmp_path/spatial_points_1.parquet"

    dsp.to_parquet(df, path, 'x', 'y', p=p, npartitions=2)
    spf = dsp.read_parquet(path)
    assert isinstance(spf, dsp.SpatialPointsFrame)
