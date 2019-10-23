import pytest
import pandas as pd
import numpy as np
import xarray as xr
from math import inf, nan
import datashader as ds
from datashader.tests.test_pandas import assert_eq_xr
import dask.dataframe as dd


def dask_DataFrame(*args, **kwargs):
    return dd.from_pandas(pd.DataFrame(*args, **kwargs), npartitions=3)


DataFrames = [pd.DataFrame, dask_DataFrame]


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_multipolygon_manual_range(DataFrame):
    df = DataFrame({
        'polygons': pd.Series([
            [0, 0, 2, 0, 2, 2, 1, 3, 0, 0,
             -inf, -inf, 1, 0.25, 1, 2, 1.75, .25, 0.25, 0.25,
             inf, inf, 2.5, 1, 4, 1, 4, 2, 2.5, 2, 2.5, 1
             ],
        ], dtype='Polygons[float64]'),
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
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])

    assert_eq_xr(agg, out)


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_multiple_polygons_auto_range(DataFrame):
    df = DataFrame({
        'polygons': pd.Series([
            [0, 0, 2, 0, 2, 2, 1, 3, 0, 0,
             -inf, -inf, 1, 0.25, 1, 2, 1.75, .25, 0.25, 0.25,
             inf, inf, 2.5, 1, 4, 1, 4, 2, 2.5, 2, 2.5, 1
             ],
        ], dtype='Polygons[float64]'),
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
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
    ], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])

    assert_eq_xr(agg, out)


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_no_overlap(DataFrame):
    df = DataFrame({
        'polygons': pd.Series([
            [1, 1, 2, 2, 1, 3, 0, 2, 1, 1,
             -inf, -inf,
             0.5, 1.5, 0.5, 2.5, 1.5, 2.5, 1.5, 1.5, 0.5, 1.5],
            [0.5, 1.5, 1.5, 1.5, 1.5, 2.5, 0.5, 2.5, 0.5, 1.5],
            [0, 1, 2, 1, 2, 3, 0, 3, 0, 1,
             1, 1, 0, 2, 1, 3, 2, 2, 1, 1]
        ], dtype='Polygons[float64]'),
    })

    cvs = ds.Canvas(plot_width=16, plot_height=16)
    agg = cvs.polygons(df, geometry='polygons', agg=ds.count())

    axis = ds.core.LinearAxis()
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((0, 2), 16), 16)
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((1, 3), 16), 16)

    sol = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])

    assert_eq_xr(agg, out)


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_no_overlap_agg(DataFrame):
    df = DataFrame({
        'polygons': pd.Series([
            [1, 1, 2, 2, 1, 3, 0, 2, 1, 1,
             -inf, -inf,
             0.5, 1.5, 0.5, 2.5, 1.5, 2.5, 1.5, 1.5, 0.5, 1.5],
            [0.5, 1.5, 1.5, 1.5, 1.5, 2.5, 0.5, 2.5, 0.5, 1.5],
            [0, 1, 2, 1, 2, 3, 0, 3, 0, 1,
             1, 1, 0, 2, 1, 3, 2, 2, 1, 1]
        ], dtype='Polygons[float64]'),
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
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        [nan,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  2.,  2.,  2., 2.,  2.,  2.],
        [nan,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,  0.,  2.,  2., 2.,  2.,  2.],
        [nan,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  2., 2.,  2.,  2.],
        [nan,  2.,  2.,  2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 2.,  2.,  2.],
        [nan,  2.,  2.,  2.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 0.,  2.,  2.],
        [nan,  2.,  2.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 0.,  0.,  2.],
        [nan,  2.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 0.,  0.,  0.],
        [nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 0.,  0.,  0.],
        [nan,  2.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 0.,  0.,  0.],
        [nan,  2.,  2.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 0.,  0.,  2.],
        [nan,  2.,  2.,  2.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 0.,  2.,  2.],
        [nan,  2.,  2.,  2.,  2.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 2.,  2.,  2.],
        [nan,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  2., 2.,  2.,  2.],
        [nan,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,  0.,  2.,  2., 2.,  2.,  2.],
        [nan,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  2.,  2.,  2., 2.,  2.,  2.]
    ])

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])
    assert_eq_xr(agg, out)
