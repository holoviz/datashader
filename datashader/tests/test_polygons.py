import pytest
import pandas as pd
import numpy as np
import xarray as xr
import datashader as ds
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
import dask.dataframe as dd

try:
    # Import to register extension arrays
    import spatialpandas  # noqa (register EAs)
    from spatialpandas import GeoDataFrame
    from spatialpandas.geometry import MultiPolygonArray
except ImportError:
    spatialpandas = None
    GeoDataFrame = None
    MultiPolygonArray = None


def dask_GeoDataFrame(*args, **kwargs):
    return dd.from_pandas(GeoDataFrame(*args, **kwargs), npartitions=3)


DataFrames = [GeoDataFrame, dask_GeoDataFrame]


@pytest.mark.skipif(not spatialpandas, reason="spatialpandas not installed")
@pytest.mark.parametrize('DataFrame', DataFrames)
def test_multipolygon_manual_range(DataFrame):
    df = DataFrame({
        'polygons': pd.Series([[
            [
                [0, 0, 2, 0, 2, 2, 1, 3, 0, 0],
                [1, 0.25, 1, 2, 1.75, .25, 0.25, 0.25]
            ], [
                [2.5, 1, 4, 1, 4, 2, 2.5, 2, 2.5, 1]
            ],
        ]], dtype='MultiPolygon[float64]'),
        'v': [1]
    })

    cvs = ds.Canvas(plot_width=16, plot_height=16)
    agg = cvs.polygons(df, geometry='polygons', agg=ds.count())

    axis = ds.core.LinearAxis()
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((0., 4.), 16), 16)
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((0., 3.), 16), 16)

    sol = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])

    assert_eq_xr(agg, out)

    assert_eq_ndarray(agg.x_range, (0, 4), close=True)
    assert_eq_ndarray(agg.y_range, (0, 3), close=True)


@pytest.mark.skipif(not spatialpandas, reason="spatialpandas not installed")
@pytest.mark.parametrize('DataFrame', DataFrames)
def test_multiple_polygons_auto_range(DataFrame):
    df = DataFrame({
        'polygons': pd.Series([[
            [
                [0, 0, 2, 0, 2, 2, 1, 3, 0, 0],
                [1, 0.25, 1, 2, 1.75, .25, 0.25, 0.25]
            ], [
                [2.5, 1, 4, 1, 4, 2, 2.5, 2, 2.5, 1]
            ],
        ]], dtype='MultiPolygon[float64]'),
        'v': [1]
    })

    cvs = ds.Canvas(plot_width=16, plot_height=16,
                    x_range=[-1, 3.5], y_range=[0.1, 2])
    agg = cvs.polygons(df, geometry='polygons', agg=ds.count())

    axis = ds.core.LinearAxis()
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-1, 3.5), 16), 16)
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((0.1, 2), 16), 16)

    sol = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    ], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])

    assert_eq_xr(agg, out)
    
    assert_eq_ndarray(agg.x_range, (-1, 3.5), close=True)
    assert_eq_ndarray(agg.y_range, (0.1, 2), close=True)


@pytest.mark.skipif(not spatialpandas, reason="spatialpandas not installed")
@pytest.mark.parametrize('DataFrame', DataFrames)
def test_no_overlap(DataFrame):
    df = DataFrame({
        'polygons': pd.Series([
            [
                [1, 1, 2, 2, 1, 3, 0, 2, 1, 1],
                [0.5, 1.5, 0.5, 2.5, 1.5, 2.5, 1.5, 1.5, 0.5, 1.5]
            ], [
                [0.5, 1.5, 1.5, 1.5, 1.5, 2.5, 0.5, 2.5, 0.5, 1.5]
            ], [
                [0, 1, 2, 1, 2, 3, 0, 3, 0, 1, 1, 1, 0, 2, 1, 3, 2, 2, 1, 1]
            ]
        ], dtype='Polygon[float64]'),
    })

    cvs = ds.Canvas(plot_width=16, plot_height=16)
    agg = cvs.polygons(df, geometry='polygons', agg=ds.count())

    axis = ds.core.LinearAxis()
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((0, 2), 16), 16)
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((1, 3), 16), 16)

    sol = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])

    assert_eq_xr(agg, out)


@pytest.mark.skipif(not spatialpandas, reason="spatialpandas not installed")
@pytest.mark.parametrize('DataFrame', DataFrames)
def test_no_overlap_agg(DataFrame):
    df = DataFrame({
        'polygons': pd.Series([
            [[1, 1, 2, 2, 1, 3, 0, 2, 1, 1],
             [0.5, 1.5, 0.5, 2.5, 1.5, 2.5, 1.5, 1.5, 0.5, 1.5]],
            [[0.5, 1.5, 1.5, 1.5, 1.5, 2.5, 0.5, 2.5, 0.5, 1.5]],
            [[0, 1, 2, 1, 2, 3, 0, 3, 0, 1, 1, 1, 0, 2, 1, 3, 2, 2, 1, 1]]
        ], dtype='Polygon[float64]'),
        'v': range(3)
    })

    cvs = ds.Canvas(plot_width=16, plot_height=16)
    agg = cvs.polygons(df, geometry='polygons', agg=ds.sum('v'))

    axis = ds.core.LinearAxis()
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((0, 2), 16), 16)
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((1, 3), 16), 16)

    sol = np.array([
        [2., 2., 2., 2., 2., 2., 2., 2., 0., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 2., 0., 0., 0., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2.],
        [2., 2., 2., 2., 1., 1., 1., 1., 1., 1., 1., 1., 0., 2., 2., 2.],
        [2., 2., 2., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 2., 2.],
        [2., 2., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 2.],
        [2., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
        [2., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
        [2., 2., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 2.],
        [2., 2., 2., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 2., 2.],
        [2., 2., 2., 2., 1., 1., 1., 1., 1., 1., 1., 1., 0., 2., 2., 2.],
        [2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 2., 0., 0., 0., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 2., 2., 0., 2., 2., 2., 2., 2., 2., 2.]
    ])

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.skipif(not spatialpandas, reason="spatialpandas not installed")
@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('scale', [4, 100])
def test_multipolygon_subpixel_vertical(DataFrame, scale):
    df = GeoDataFrame({
        'geometry': MultiPolygonArray([[
            [[0, 0, 1, 0, 1, 1, 0, 1, 0, 0]],
            [[2, 0, 3, 0, 3, 1, 2, 1, 2, 0]],
        ]])
    })

    cvs = ds.Canvas(
        plot_height=8, plot_width=8,
        x_range=(0, 4),
        y_range=(-2 * scale, 2 * scale)
    )
    agg = cvs.polygons(df, 'geometry', agg=ds.count())

    sol = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int32)

    axis = ds.core.LinearAxis()
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((0, 4), 8), 8)
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-2 * scale, 2 * scale), 8), 8)
    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.skipif(not spatialpandas, reason="spatialpandas not installed")
@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('scale', [4, 100])
def test_multipolygon_subpixel_horizontal(DataFrame, scale):
    df = GeoDataFrame({
        'geometry': MultiPolygonArray([[
            [[0, 0, 1, 0, 1, 1, 0, 1, 0, 0]],
            [[0, 2, 1, 2, 1, 3, 0, 3, 0, 2]],
        ]])
    })

    cvs = ds.Canvas(
        plot_height=8, plot_width=8,
        x_range=(-2 * scale, 2 * scale),
        y_range=(0, 4)
    )
    agg = cvs.polygons(df, 'geometry', agg=ds.count())

    sol = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int32)

    axis = ds.core.LinearAxis()
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-2 * scale, 2 * scale), 8), 8)
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((0, 4), 8), 8)
    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.skipif(not spatialpandas, reason="spatialpandas not installed")
def test_spatial_index_not_dropped():
    # Issue 1121
    df = GeoDataFrame({
        'some_geom': MultiPolygonArray([
            [[[0, 0, 1, 0, 1, 1, 0, 1, 0, 0]]],
            [[[0, 2, 1, 2, 1, 3, 0, 3, 0, 2]]],
        ]),
        'other': [23, 45],  # This column is not used and will be dropped.
    })

    assert df.some_geom.array._sindex is None
    sindex = df.some_geom.array.sindex
    assert sindex is not None

    glyph = ds.glyphs.polygon.PolygonGeom('some_geom')
    agg = ds.count()

    df2, _ = ds.core._bypixel_sanitise(df, glyph, agg)

    assert df2.columns == ['some_geom']
    assert df2.some_geom.array._sindex == df.some_geom.array._sindex
