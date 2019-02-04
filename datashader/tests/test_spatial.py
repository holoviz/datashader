import os
import pytest
import numpy as np
import pandas as pd

from datashader.spatial import SpatialPointsFrame


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


@pytest.fixture()
def s_points_frame(tmp_path, df):

    p = 5
    filename = os.path.join(tmp_path, 'spatial_points.parquet')

    SpatialPointsFrame.partition_and_write(
        df, 'x', 'y', filename=filename, p=p, npartitions=10)

    return SpatialPointsFrame(filename=filename)


def test_spacial_points_frame_properties(s_points_frame):
    assert s_points_frame.x == 'x'
    assert s_points_frame.y == 'y'
    assert s_points_frame.p == 5
    assert s_points_frame.npartitions == 10
    assert s_points_frame.x_range == (0, 1)
    assert s_points_frame.y_range == (0, 2)

    # x_bin_edges
    np.testing.assert_array_equal(
        s_points_frame.x_bin_edges,
        np.linspace(0.0, 1.0, 2 ** 5 + 1))

    # y_bin_edges
    np.testing.assert_array_equal(
        s_points_frame.y_bin_edges,
        np.linspace(0.0, 2.0, 2 ** 5 + 1))

    # distance_divisions
    distance_divisions = s_points_frame.distance_divisions
    assert len(distance_divisions) == 10 + 1


@pytest.mark.parametrize('x_range,y_range', [
    ((0, 0.2), (0, 0.2)),
    ((0.3, 1.0), (0.5, 1.5)),
    ((5, 10), (5, 10))  # Outside of bounds
])
def test_query_partitions(s_points_frame, x_range, y_range):

    # Get original
    ddf = s_points_frame.frame

    # Query subset
    query_ddf = s_points_frame.query_partitions(x_range, y_range)

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


def test_validate_parquet_file(df, tmp_path):

    # Write DataFrame to parquet
    filename = os.path.join(tmp_path, 'df.parquet')
    df.to_parquet(filename, engine='fastparquet')

    # Try to construct a SpatialPointsFrame from it
    with pytest.raises(ValueError) as e:
        SpatialPointsFrame(filename)

    assert 'SpatialPointsFrame.partition_and_write' in str(e.value)

