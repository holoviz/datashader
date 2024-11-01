# Testing GeoPandas and SpatialPandas
import contextlib

import datashader as ds
from datashader.tests.test_pandas import assert_eq_ndarray
import numpy as np
from numpy import nan
import pytest
from datashader.tests.utils import dask_switcher
from packaging.version import Version

try:
    import dask.dataframe as dd
except ImportError:
    dd = None

_backends = [
    pytest.param(False, id="dask"),
]

_extras = ["spatialpandas.dask", "dask_geopandas.backends", "dask_geopandas"]

with contextlib.suppress(ImportError):
    import dask_geopandas

    if Version(dask_geopandas.__version__) >= Version("0.4.0"):
        _backends.append(pytest.param(True, id="dask-expr"))


@pytest.fixture(params=_backends)
def dask_both(request):
    with dask_switcher(query=request.param, extras=_extras): ...
    return request.param

@pytest.fixture
def dask_classic(request):
    with dask_switcher(query=False, extras=_extras): ...

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

nybb_points_sol = np.array([
    [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 3, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 7, 6, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 4, 3, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 2, 4, 2, 8, 3, 0, 1, 2, 2, 0, 3, 1, 0, 0, 0, 2, 1],
    [0, 0, 1, 0, 5, 2, 3, 0, 0, 2, 4, 2, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 2, 3, 8, 3, 0, 1, 5, 2, 7, 5, 0, 3, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 5, 5, 3, 4, 3, 3, 1, 2, 0, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 5, 2, 4, 3, 3, 2, 3, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 2, 5, 4, 1, 3, 1, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 5, 2, 2, 5, 2, 3, 1, 1, 2, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 6, 1, 2, 2, 3, 2, 1, 2, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 8, 7, 1, 2, 2, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 9, 6, 3, 3, 1, 4, 1, 3, 5, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 1, 0, 0, 2, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 4, 7, 4, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 8, 9, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 4, 4, 5, 5, 5, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 5, 7, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5, 0, 0, 0],
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


@pytest.mark.skipif(not dask_geopandas, reason="dask_geopandas not installed")
def test_dask_geopandas_switcher(dask_both):
    import dask_geopandas
    if dask_both:
        assert dask_geopandas.expr.GeoDataFrame == dask_geopandas.GeoDataFrame
    else:
        assert dask_geopandas.core.GeoDataFrame == dask_geopandas.GeoDataFrame


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
    df = geopandas.read_file(geodatasets.get_path("nybb"))
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
def test_lines_dask_geopandas(geom_type, explode, use_boundary, npartitions, dask_both):
    df = geopandas.read_file(geodatasets.get_path("nybb"))
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
def test_lines_spatialpandas(geom_type, explode, use_boundary, npartitions, dask_classic):
    df = geopandas.read_file(geodatasets.get_path("nybb"))
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
    df = geopandas.read_file(geodatasets.get_path("nybb"))

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
def test_points_dask_geopandas(geom_type, npartitions, dask_both):
    df = geopandas.read_file(geodatasets.get_path("nybb"))

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
def test_points_spatialpandas(geom_type, npartitions, dask_classic):
    df = geopandas.read_file(geodatasets.get_path("nybb"))

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
    df = geopandas.read_file(geodatasets.get_path("nybb"))
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
def test_polygons_dask_geopandas(geom_type, npartitions, dask_both):
    df = geopandas.read_file(geodatasets.get_path("nybb"))
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
def test_polygons_spatialpandas(geom_type, npartitions, dask_classic):
    df = geopandas.read_file(geodatasets.get_path("nybb"))
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
