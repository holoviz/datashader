# Testing GeoPandas and SpatialPandas

from importlib.util import find_spec

import datashader as ds
from datashader.tests.test_pandas import assert_eq_ndarray
import numpy as np
from numpy import nan
import pytest

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


try:
    import dask_geopandas
except ImportError:
    dask_geopandas = None

try:
    import geodatasets
except ImportError:
    geodatasets = None

try:
    import geopandas
except ImportError:
    geopandas = None

try:
    # Import to register extension arrays
    import spatialpandas  # noqa (register EAs)
except ImportError:
    spatialpandas = None


nybb_lines_sol = np.array([
    [ 0.,  0.,  0., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],  # noqa: E501
    [ 0., nan,  0.,  0.,  0., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],  # noqa: E501
    [ 0.,  0., nan, nan,  0.,  0., nan, nan, nan, nan, nan,  1.,  1.,  1., nan, nan, nan, nan, nan, nan],  # noqa: E501
    [nan,  0., nan, nan, nan,  0.,  0.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  1., nan, nan, nan, nan],  # noqa: E501
    [nan,  0.,  0., nan, nan, nan,  0.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  1.,  1.,  1., nan],  # noqa: E501
    [nan,  0.,  0., nan, nan, nan,  0.,  2.,  2., nan, nan,  2.,  2.,  2.,  2.,  2.,  1.,  1.,  1., nan],  # noqa: E501
    [nan,  0.,  0.,  0.,  0.,  0.,  0.,  2., nan, nan, nan, nan,  2.,  2.,  2.,  2.,  1.,  1.,  1., nan],  # noqa: E501
    [nan, nan,  0.,  0., nan,  0.,  0.,  2.,  2.,  2., nan, nan, nan,  2.,  2.,  1.,  1., nan,  1.,  1.],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan,  3.,  2., nan, nan,  2.,  2.,  2., nan, nan, nan,  1.,  1.],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan,  3.,  3.,  3.,  2., nan,  2.,  2.,  2., nan, nan, nan, nan,  1.],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan,  3.,  3.,  3.,  2.,  2., nan, nan, nan, nan, nan,  1.,  1.],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan,  3., nan,  3.,  2., nan, nan, nan, nan, nan, nan, nan,  1.],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan,  3.,  3.,  3.,  3., nan, nan,  1., nan, nan,  1.,  1.,  1.],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan,  3., nan,  3.,  3.,  4.,  1.,  1.,  1.,  1.,  1., nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,  3.,  4.,  4.,  4.,  4.,  4.,  4.,  1., nan, nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,  4., nan,  4.,  4.,  4.,  4., nan, nan, nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,  4., nan, nan,  4.,  4.,  4.,  4., nan, nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  4.,  4., nan, nan,  4.,  4.,  4., nan, nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  4.,  4., nan, nan,  4.,  4.,  4., nan, nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  4.,  4.,  4., nan, nan, nan, nan, nan],  # noqa: E501
])

# This data is for checking repeatability in our tests, not correctness.
# GeoPandas 1.1.0 updated the sampling of points in https://github.com/geopandas/geopandas/pull/3471
nybb_points_sol = np.array([
    [2,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [3,  4,  1,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  3,  6,  6,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  4,  4,  6,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  1,  2,  3,  5,  4,  3,  3,  0,  2,  1,  1,  0,  3,  0,  0,  0,  3,  2,  0],
    [0,  0,  3,  2,  5,  2,  2,  0,  1,  2,  1,  5,  0,  1,  0,  0,  1,  0,  0,  0],
    [0,  0,  2,  1,  3,  7,  0,  0,  1,  5,  7,  6,  0,  0,  1,  0,  0,  0,  1,  0],
    [0,  0,  0,  0,  0,  0,  1,  0,  1,  6,  4,  6,  3,  3,  1,  2,  1,  2,  5,  2],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  5,  2,  3,  4,  2,  5,  5,  1,  2,  2,  0],
    [0,  0,  0,  0,  0,  0,  0,  1,  1,  2,  3,  2,  3,  2,  4,  0,  4,  0,  3,  2],
    [0,  0,  0,  0,  0,  0,  0,  0,  0, 11,  3,  2,  2,  4,  1,  4,  1,  2,  0,  4],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  4,  3,  1,  3,  1,  1,  2,  0,  4,  1],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  5, 10,  1,  0,  1,  1,  2,  1,  1,  3,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 14,  7,  1,  1,  5,  2,  2,  3,  3,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  7,  2,  0,  0,  1,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  7,  6,  5,  2,  2,  1,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3,  1,  8, 12,  7,  1,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  5,  4,  8,  2,  7,  2,  4,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  5,  4,  4,  4,  2,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3,  2,  3,  1,  0,  0,  0],
], dtype=np.uint32)


nybb_polygons_sol = np.array([
    [ 0.,  0., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],  # noqa: E501
    [ 0.,  0.,  0., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],  # noqa: E501
    [nan,  0.,  0.,  0.,  0., nan, nan, nan, nan, nan, nan,  1., nan, nan, nan, nan, nan, nan, nan, nan],  # noqa: E501
    [nan, nan,  0.,  0.,  0.,  0., nan,  0., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],  # noqa: E501
    [nan, nan,  0.,  0.,  0.,  0.,  0.,  0., nan,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  1., nan, nan, nan],  # noqa: E501
    [nan, nan,  0.,  0.,  0.,  0.,  0., nan,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  1., nan, nan],  # noqa: E501
    [nan, nan,  0.,  0.,  0.,  0.,  0., nan,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  1., nan, nan],  # noqa: E501
    [nan, nan, nan,  0., nan, nan, nan, nan,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  1.,  1.,  1.,  1., nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan,  3.,  2.,  2.,  2.,  2.,  2.,  1.,  1.,  1.,  1.,  1., nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan,  3.,  3.,  3.,  2.,  2.,  2.,  1.,  1.,  1.,  1.,  1.,  1., nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,  3.,  2.,  1.,  1.,  1.,  1.,  1.,  1.,  1., nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,  3.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,  3.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,  3.,  1.,  4., nan,  1.,  1.,  1., nan, nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,  3.,  4.,  4., nan, nan, nan, nan, nan, nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,  4.,  4.,  4.,  4., nan, nan, nan, nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,  4.,  4.,  4.,  4.,  4.,  4., nan, nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  4.,  4.,  4.,  4.,  4.,  4., nan, nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,  4.,  4.,  4.,  4.,  4.,  4., nan, nan],  # noqa: E501
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  4.,  4.,  4., nan, nan, nan, nan, nan],  # noqa: E501
])


def _nybb_data():
    if find_spec("pyogrio") or find_spec("fiona"):
        return geopandas.read_file(geodatasets.get_path("nybb"))
    else:
        pytest.skip("Neither pyogrio nor fiona is installed")


@pytest.mark.skipif(not geodatasets, reason="geodatasets not installed")
@pytest.mark.skipif(not geopandas, reason="geopandas not installed")
@pytest.mark.parametrize("geom_type, explode, use_boundary",
    [
        ("multipolygon", False, False),
        ("polygon", True, False),
        ("multilinestring", False, True),
        ("linestring", True, True),
    ],
)
def test_lines_geopandas(geom_type, explode, use_boundary):
    df = _nybb_data()
    df["col"] = np.arange(len(df))  # Extra column for aggregation.
    geometry = "boundary" if use_boundary else "geometry"

    if explode:
        df = df.explode(index_parts=False)  # Multipolygon -> polygon.
    if use_boundary:
        df["boundary"] = df.boundary
    unique_geom_type = df[geometry].geom_type.unique()
    assert len(unique_geom_type) == 1 and unique_geom_type[0].lower() == geom_type

    canvas = ds.Canvas(plot_height=20, plot_width=20)
    agg = canvas.line(source=df, geometry=geometry, agg=ds.max("col"))
    assert_eq_ndarray(agg.data, nybb_lines_sol)


@pytest.mark.skipif(not geodatasets, reason="geodatasets not installed")
@pytest.mark.skipif(not dask_geopandas, reason="dask_geopandas not installed")
@pytest.mark.skipif(not geopandas, reason="geopandas not installed")
@pytest.mark.parametrize('npartitions', [1, 2, 5])
@pytest.mark.parametrize("geom_type, explode, use_boundary",
    [
        ("multipolygon", False, False),
        ("polygon", True, False),
        ("multilinestring", False, True),
        ("linestring", True, True),
    ],
)
def test_lines_dask_geopandas(geom_type, explode, use_boundary, npartitions):
    df = _nybb_data()
    df["col"] = np.arange(len(df))  # Extra column for aggregation.
    geometry = "boundary" if use_boundary else "geometry"

    if explode:
        df = df.explode(index_parts=False)  # Multipolygon -> polygon.
    if use_boundary:
        df["boundary"] = df.boundary
    unique_geom_type = df[geometry].geom_type.unique()
    assert len(unique_geom_type) == 1 and unique_geom_type[0].lower() == geom_type

    df = dd.from_pandas(df, npartitions=npartitions)
    assert df.npartitions == npartitions
    df.calculate_spatial_partitions()

    canvas = ds.Canvas(plot_height=20, plot_width=20)
    agg = canvas.line(source=df, geometry=geometry, agg=ds.max("col"))
    assert_eq_ndarray(agg.data, nybb_lines_sol)


@pytest.mark.skipif(not geodatasets, reason="geodatasets not installed")
@pytest.mark.skipif(not spatialpandas, reason="spatialpandas not installed")
@pytest.mark.parametrize('npartitions', [1, 2, 5])
@pytest.mark.parametrize("geom_type, explode, use_boundary",
    [
        ("multipolygon", False, False),
        ("polygon", True, False),
        ("multilinestring", False, True),
        ("linestring", True, True),
    ],
)
def test_lines_spatialpandas(geom_type, explode, use_boundary, npartitions):
    df = _nybb_data()
    df["col"] = np.arange(len(df))  # Extra column for aggregation.
    geometry = "boundary" if use_boundary else "geometry"

    if explode:
        df = df.explode(index_parts=False)  # Multipolygon -> polygon.
    if use_boundary:
        df["boundary"] = df.boundary
    unique_geom_type = df[geometry].geom_type.unique()
    assert len(unique_geom_type) == 1 and unique_geom_type[0].lower() == geom_type

    df = spatialpandas.GeoDataFrame(df)
    if npartitions > 0:
        df = dd.from_pandas(df, npartitions=npartitions)
        assert df.npartitions == npartitions

    canvas = ds.Canvas(plot_height=20, plot_width=20)
    agg = canvas.line(source=df, geometry=geometry, agg=ds.max("col"))
    assert_eq_ndarray(agg.data, nybb_lines_sol)


@pytest.mark.skipif(not geodatasets, reason="geodatasets not installed")
@pytest.mark.skipif(not geopandas, reason="geopandas not installed")
@pytest.mark.parametrize("geom_type", ["multipoint", "point"])
def test_points_geopandas(geom_type):
    df = _nybb_data()

    df["geometry"] = df["geometry"].sample_points(100, rng=93814)  # multipoint
    if geom_type == "point":
        df = df.explode(index_parts=False)  # Multipoint -> point.
    unique_geom_type = df["geometry"].geom_type.unique()
    assert len(unique_geom_type) == 1 and unique_geom_type[0].lower() == geom_type

    canvas = ds.Canvas(plot_height=20, plot_width=20)
    agg = canvas.points(source=df, geometry="geometry", agg=ds.count())
    assert_eq_ndarray(agg.data, nybb_points_sol)


@pytest.mark.skipif(not geodatasets, reason="geodatasets not installed")
@pytest.mark.skipif(not geopandas, reason="geopandas not installed")
@pytest.mark.parametrize('npartitions', [1, 2, 5])
@pytest.mark.parametrize("geom_type", ["multipoint", "point"])
def test_points_dask_geopandas(geom_type, npartitions):
    df = _nybb_data()

    df["geometry"] = df["geometry"].sample_points(100, rng=93814)  # multipoint
    if geom_type == "point":
        df = df.explode(index_parts=False)  # Multipoint -> point.
    unique_geom_type = df["geometry"].geom_type.unique()
    assert len(unique_geom_type) == 1 and unique_geom_type[0].lower() == geom_type

    df = dd.from_pandas(df, npartitions=npartitions)
    assert df.npartitions == npartitions
    df.calculate_spatial_partitions()

    canvas = ds.Canvas(plot_height=20, plot_width=20)
    agg = canvas.points(source=df, geometry="geometry", agg=ds.count())
    assert_eq_ndarray(agg.data, nybb_points_sol)


@pytest.mark.skipif(not geodatasets, reason="geodatasets not installed")
@pytest.mark.skipif(not spatialpandas, reason="spatialpandas not installed")
@pytest.mark.parametrize('npartitions', [0, 1, 2, 5])
@pytest.mark.parametrize("geom_type", ["multipoint", "point"])
def test_points_spatialpandas(geom_type, npartitions):
    df = _nybb_data()

    df["geometry"] = df["geometry"].sample_points(100, rng=93814)  # multipoint
    if geom_type == "point":
        df = df.explode(index_parts=False)  # Multipoint -> point.
    unique_geom_type = df["geometry"].geom_type.unique()
    assert len(unique_geom_type) == 1 and unique_geom_type[0].lower() == geom_type

    df = spatialpandas.GeoDataFrame(df)
    if npartitions > 0:
        df = dd.from_pandas(df, npartitions=npartitions)
        assert df.npartitions == npartitions

    canvas = ds.Canvas(plot_height=20, plot_width=20)
    agg = canvas.points(source=df, geometry="geometry", agg=ds.count())
    assert_eq_ndarray(agg.data, nybb_points_sol)


@pytest.mark.skipif(not geodatasets, reason="geodatasets not installed")
@pytest.mark.skipif(not geopandas, reason="geopandas not installed")
@pytest.mark.parametrize("geom_type", ["multipolygon", "polygon"])
def test_polygons_geopandas(geom_type):
    df = _nybb_data()
    df["col"] = np.arange(len(df))  # Extra column for aggregation.

    if geom_type == "polygon":
        df = df.explode(index_parts=False)  # Multipolygon -> polygon.
    unique_geom_type = df["geometry"].geom_type.unique()
    assert len(unique_geom_type) == 1 and unique_geom_type[0].lower() == geom_type

    canvas = ds.Canvas(plot_height=20, plot_width=20)
    agg = canvas.polygons(source=df, geometry="geometry", agg=ds.max("col"))
    assert_eq_ndarray(agg.data, nybb_polygons_sol)


@pytest.mark.skipif(not geodatasets, reason="geodatasets not installed")
@pytest.mark.skipif(not dask_geopandas, reason="dask_geopandas not installed")
@pytest.mark.skipif(not geopandas, reason="geopandas not installed")
@pytest.mark.parametrize('npartitions', [1, 2, 5])
@pytest.mark.parametrize("geom_type", ["multipolygon", "polygon"])
def test_polygons_dask_geopandas(geom_type, npartitions):
    df = _nybb_data()
    df["col"] = np.arange(len(df))

    if geom_type == "polygon":
        df = df.explode(index_parts=False)  # Multipolygon -> polygon.
    unique_geom_type = df["geometry"].geom_type.unique()
    assert len(unique_geom_type) == 1 and unique_geom_type[0].lower() == geom_type

    df = dd.from_pandas(df, npartitions=npartitions)
    assert df.npartitions == npartitions
    df.calculate_spatial_partitions()

    canvas = ds.Canvas(plot_height=20, plot_width=20)
    agg = canvas.polygons(source=df, geometry="geometry", agg=ds.max("col"))
    assert_eq_ndarray(agg.data, nybb_polygons_sol)


@pytest.mark.skipif(not geodatasets, reason="geodatasets not installed")
@pytest.mark.skipif(not geopandas, reason="geopandas not installed")
@pytest.mark.skipif(not spatialpandas, reason="spatialpandas not installed")
@pytest.mark.parametrize('npartitions', [0, 1, 2, 5])
@pytest.mark.parametrize("geom_type", ["multipolygon", "polygon"])
def test_polygons_spatialpandas(geom_type, npartitions):
    df = _nybb_data()
    df["col"] = np.arange(len(df))

    if geom_type == "polygon":
        df = df.explode(index_parts=False)  # Multipolygon -> polygon.
    unique_geom_type = df["geometry"].geom_type.unique()
    assert len(unique_geom_type) == 1 and unique_geom_type[0].lower() == geom_type

    df = spatialpandas.GeoDataFrame(df)
    if npartitions > 0:
        df = dd.from_pandas(df, npartitions=npartitions)
        assert df.npartitions == npartitions

    canvas = ds.Canvas(plot_height=20, plot_width=20)
    agg = canvas.polygons(source=df, geometry="geometry", agg=ds.max("col"))
    assert_eq_ndarray(agg.data, nybb_polygons_sol)
