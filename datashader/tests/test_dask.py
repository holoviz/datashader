from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from numpy import nan
from packaging.version import Version

import datashader as ds
from datashader.datatypes import RaggedArray
import datashader.utils as du

import pytest
from datashader.tests.utils import dask_switcher
from datashader.tests.test_pandas import _pandas

try:
    import spatialpandas as sp
    import spatialpandas.dask  # noqa (API import)
except ImportError:
    sp = None

from datashader.tests.test_pandas import (
    assert_eq_xr, assert_eq_ndarray, values
)


try:
    import dask.dataframe as dd
    from dask.context import config
    config.set(scheduler='synchronous')
except ImportError:
    pytestmark = pytest.importorskip("dask")



@dask_switcher(query=False)
def _dask():
    return dd.from_pandas(_pandas(), npartitions=2)

@dask_switcher(query=True)
def _dask_expr():
    return dd.from_pandas(_pandas(), npartitions=2)

@dask_switcher(query=False, extras=["dask_cudf"])
def _dask_cudf():
    import dask_cudf
    _dask = dd.from_pandas(_pandas(), npartitions=2)
    if Version(dask_cudf.__version__) >= Version("24.06"):
        return _dask.to_backend("cudf")
    else:
        return dask_cudf.from_dask_dataframe(_dask)

@dask_switcher(query=True, extras=["dask_cudf"])
def _dask_expr_cudf():
    import dask_cudf
    if Version(dask_cudf.__version__) < Version("24.06"):
        pytest.skip("dask-expr requires dask-cudf 24.06 or later")
    _dask = dd.from_pandas(_pandas(), npartitions=2)
    return _dask.to_backend("cudf")

_backends = [
    pytest.param(_dask, id="dask"),
    pytest.param(_dask_expr, id="dask-expr"),
    pytest.param(_dask_cudf, marks=pytest.mark.gpu, id="dask-cudf"),
    pytest.param(_dask_expr_cudf, marks=pytest.mark.gpu, id="dask-expr-cudf"),
]

@pytest.fixture(params=_backends)
def ddf(request):
    return request.param()


@pytest.fixture(params=[1, 2, 4])
def npartitions(request):
    return request.param

@dask_switcher(query=False)
def _dask_DataFrame(*args, **kwargs):
    if kwargs.pop("geo", False):
        df = sp.GeoDataFrame(*args, **kwargs)
    else:
        df = pd.DataFrame(*args, **kwargs)
    return dd.from_pandas(df, npartitions=2)


@dask_switcher(query=True)
def _dask_expr_DataFrame(*args, **kwargs):
    if kwargs.pop("geo", False):
        pytest.skip("dask-expr currently does not work with spatialpandas")
        # df = sp.GeoDataFrame(*args, **kwargs)
    else:
        df = pd.DataFrame(*args, **kwargs)
    return dd.from_pandas(df, npartitions=2)


@dask_switcher(query=False, extras=["dask_cudf"])
def _dask_cudf_DataFrame(*args, **kwargs):
    import cudf
    import dask_cudf
    if kwargs.pop("geo", False):
        # As of dask-cudf version 24.06, dask-cudf is not
        # compatible with spatialpandas version 0.4.10
        pytest.skip("dask-cudf currently does not work with spatialpandas")
    cdf = cudf.DataFrame.from_pandas(
        pd.DataFrame(*args, **kwargs), nan_as_null=False
    )
    return dask_cudf.from_cudf(cdf, npartitions=2)


@dask_switcher(query=True, extras=["dask_cudf"])
def _dask_expr_cudf_DataFrame(*args, **kwargs):
    import cudf
    import dask_cudf

    if Version(dask_cudf.__version__) < Version("24.06"):
        pytest.skip("dask-expr requires dask-cudf 24.06 or later")

    if kwargs.pop("geo", False):
        # As of dask-cudf version 24.06, dask-cudf is not
        # compatible with spatialpandas version 0.4.10
        pytest.skip("dask-cudf currently does not work with spatialpandas")
    cdf = cudf.DataFrame.from_pandas(
        pd.DataFrame(*args, **kwargs), nan_as_null=False
    )
    return dask_cudf.from_cudf(cdf, npartitions=2)


_backends = [
    pytest.param(_dask_DataFrame, id="dask"),
    pytest.param(_dask_expr_DataFrame, id="dask-expr"),
    pytest.param(_dask_cudf_DataFrame, marks=pytest.mark.gpu, id="dask-cudf"),
    pytest.param(_dask_expr_cudf_DataFrame, marks=pytest.mark.gpu, id="dask-expr-cudf"),
]

@pytest.fixture(params=_backends)
def DataFrame(request):
    return request.param

c = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))
c_logx = ds.Canvas(plot_width=2, plot_height=2, x_range=(1, 10),
                   y_range=(0, 1), x_axis_type='log')
c_logy = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1),
                   y_range=(1, 10), y_axis_type='log')
c_logxy = ds.Canvas(plot_width=2, plot_height=2, x_range=(1, 10),
                    y_range=(1, 10), x_axis_type='log', y_axis_type='log')

axis = ds.core.LinearAxis()
lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)
coords = [lincoords, lincoords]
dims = ['y', 'x']


def assert_eq(agg, b):
    assert agg.equals(b)


def floats(n):
    """Returns contiguous list of floats from initial point"""
    while True:
        yield n
        n = n + np.spacing(n)


@pytest.mark.gpu
def test_check_query_setting():
    import os
    from subprocess import check_output, SubprocessError

    # dask-cudf does not support query planning as of 24.04.
    # So we check that it is not set outside of Python.
    assert os.environ.get('DASK_DATAFRAME__QUERY_PLANNING', 'false').lower() != 'true'

    # This also have problem with the global setting so we check
    try:
        cmd = ['dask', 'config', 'get', 'dataframe.query-planning']
        output = check_output(cmd, text=True).strip().lower()
        assert output != 'true'
    except SubprocessError:
        # Newer version will error out if not set
        pass


def test_count(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray(np.array([[5, 5], [5, 5]], dtype='i4'),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.count('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.count('i64')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.count()), out)
    out = xr.DataArray(np.array([[4, 5], [5, 5]], dtype='i4'),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.count('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.count('f64')), out)


def test_any(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray(np.array([[True, True], [True, True]]),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.any('i64')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.any('f64')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.any()), out)
    out = xr.DataArray(np.array([[True, True], [True, False]]),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.any('empty_bin')), out)


def test_sum(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray(
        values(_pandas().i32).reshape((2, 2, 5)).sum(axis=2, dtype='f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.sum('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.sum('i64')), out)

    out = xr.DataArray(
        np.nansum(values(_pandas().f64).reshape((2, 2, 5)), axis=2).T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.sum('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.sum('f64')), out)


def test_first(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.first('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.first('i64')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.first('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.first('f64')), out)


def test_last(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.last('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.last('i64')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.last('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.last('f64')), out)


def test_min(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray(
        values(_pandas().i64).reshape((2, 2, 5)).min(axis=2).astype('f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.min('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.min('i64')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.min('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.min('f64')), out)


def test_max(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray(
        values(_pandas().i64).reshape((2, 2, 5)).max(axis=2).astype('f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.max('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.max('i64')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.max('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.max('f64')), out)


def test_min_row_index(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds._min_row_index()), out)


def test_max_row_index(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds._max_row_index()), out)


def test_min_n(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[-3, -1, 0, 4, nan, nan], [-13, -11, 10, 12, 14, nan]],
                         [[-9, -7, -5, 6, 8, nan], [-19, -17, -15, 16, 18, nan]]])
    for n in range(1, 7):
        agg = c.points(ddf, 'x', 'y', ds.min_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(ddf, 'x', 'y', ds.min('plusminus')).data)


def test_max_n(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[4, 0, -1, -3, nan, nan], [14, 12, 10, -11, -13, nan]],
                         [[8, 6, -5, -7, -9, nan], [18, 16, -15, -17, -19, nan]]])
    for n in range(1, 7):
        agg = c.points(ddf, 'x', 'y', ds.max_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(ddf, 'x', 'y', ds.max('plusminus')).data)


def test_min_n_row_index(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[0, 1, 2, 3, 4, -1], [10, 11, 12, 13, 14, -1]],
                         [[5, 6, 7, 8, 9, -1], [15, 16, 17, 18, 19, -1]]])
    for n in range(1, 7):
        agg = c.points(ddf, 'x', 'y', ds._min_n_row_index(n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(ddf, 'x', 'y', ds._min_row_index()).data)


def test_max_n_row_index(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[4, 3, 2, 1, 0, -1], [14, 13, 12, 11, 10, -1]],
                         [[9, 8, 7, 6, 5, -1], [19, 18, 17, 16, 15, -1]]])
    for n in range(1, 7):
        agg = c.points(ddf, 'x', 'y', ds._max_n_row_index(n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(ddf, 'x', 'y', ds._max_row_index()).data)


def test_first_n(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[0, -1, -3, 4, nan, nan], [10, -11, 12, -13, 14, nan]],
                         [[-5, 6, -7, 8, -9, nan], [-15, 16, -17, 18, -19, nan]]])
    for n in range(1, 7):
        agg = c.points(ddf, 'x', 'y', ds.first_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(ddf, 'x', 'y', ds.first('plusminus')).data)


def test_last_n(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[4, -3, -1, 0, nan, nan], [14, -13, 12, -11, 10, nan]],
                         [[-9, 8, -7, 6, -5, nan], [-19, 18, -17, 16, -15, nan]]])
    for n in range(1, 7):
        agg = c.points(ddf, 'x', 'y', ds.last_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(ddf, 'x', 'y', ds.last('plusminus')).data)


def test_categorical_count(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol = np.array([[[2, 1, 1, 1], [1, 1, 2, 1]], [[1, 2, 1, 1], [1, 1, 1, 2]]], dtype=np.uint32)
    assert_eq_ndarray(c.points(ddf, 'x', 'y', ds.by('cat2')).data, sol)

    # ds.summary(name=ds.by("cat2")) should give same result as ds.by("cat2"). Issue 1252
    dataset = c.points(ddf, 'x', 'y', ds.summary(name=ds.by('cat2')))
    assert_eq_ndarray(dataset["name"].data, sol)


def test_categorical_min(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_int = np.array([[[0, 1, 2, 3], [12, 13, 10, 11]], [[8, 5, 6, 7], [16, 17, 18, 15]]],
                       dtype=np.float64)
    sol_float = np.array([[[0, 1, nan, 3], [12, 13, 10, 11]], [[8, 5, 6, 7], [16, 17, 18, 15]]])
    assert_eq_ndarray(c.points(ddf, 'x', 'y', ds.by('cat2', ds.min('i32'))).data, sol_int)
    assert_eq_ndarray(c.points(ddf, 'x', 'y', ds.by('cat2', ds.min('i64'))).data, sol_int)
    assert_eq_ndarray(c.points(ddf, 'x', 'y', ds.by('cat2', ds.min('f32'))).data, sol_float)
    assert_eq_ndarray(c.points(ddf, 'x', 'y', ds.by('cat2', ds.min('f64'))).data, sol_float)


def test_categorical_max(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_int = np.array([[[4, 1, 2, 3], [12, 13, 14, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]],
                       dtype=np.float64)
    sol_float = np.array([[[4, 1, nan, 3], [12, 13, 14, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]])
    assert_eq_ndarray(c.points(ddf, 'x', 'y', ds.by('cat2', ds.max('i32'))).data, sol_int)
    assert_eq_ndarray(c.points(ddf, 'x', 'y', ds.by('cat2', ds.max('i64'))).data, sol_int)
    assert_eq_ndarray(c.points(ddf, 'x', 'y', ds.by('cat2', ds.max('f32'))).data, sol_float)
    assert_eq_ndarray(c.points(ddf, 'x', 'y', ds.by('cat2', ds.max('f64'))).data, sol_float)


def test_categorical_min_n(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[[0, 4, nan], [1, nan, nan], [nan, nan, nan], [3, nan, nan]],
                          [[12, nan, nan], [13, nan, nan], [10, 14, nan], [11, nan, nan]]],
                         [[[8, nan, nan], [5, 9, nan], [6, nan, nan], [7, nan, nan]],
                          [[16, nan, nan], [17, nan, nan], [18, nan, nan], [15, 19, nan]]]])
    for n in range(1, 3):
        agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.min_n('f32', n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data,
                              c.points(ddf, 'x', 'y', ds.by('cat2', ds.min('f32'))).data)


def test_categorical_max_n(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[[4, 0, nan], [1, nan, nan], [nan, nan, nan], [3, nan, nan]],
                          [[12, nan, nan], [13, nan, nan], [14, 10, nan], [11, nan, nan]]],
                         [[[8, nan, nan], [9, 5, nan], [6, nan, nan], [7, nan, nan]],
                          [[16, nan, nan], [17, nan, nan], [18, nan, nan], [19, 15, nan]]]])
    for n in range(1, 3):
        agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.max_n('f32', n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data,
                              c.points(ddf, 'x', 'y', ds.by('cat2', ds.max('f32'))).data)


def test_categorical_min_row_index(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[0, 1, 2, 3], [12, 13, 10, 11]], [[8, 5, 6, 7], [16, 17, 18, 15]]])
    agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds._min_row_index()))
    assert_eq_ndarray(agg.data, solution)


def test_categorical_max_row_index(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[4, 1, 2, 3], [12, 13, 14, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]])
    agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds._max_row_index()))
    assert_eq_ndarray(agg.data, solution)


def test_categorical_min_n_row_index(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[[0, 4, -1], [1, -1, -1], [2, -1, -1], [3, -1, -1]],
                          [[12, -1, -1], [13, -1, -1], [10, 14, -1], [11, -1, -1]]],
                         [[[8, -1, -1], [5, 9, -1], [6, -1, -1], [7, -1, -1]],
                          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [15, 19, -1]]]])
    for n in range(1, 3):
        agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds._min_n_row_index(n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data,
                              c.points(ddf, 'x', 'y', ds.by('cat2', ds._min_row_index())).data)


def test_categorical_max_n_row_index(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[[4, 0, -1], [1, -1, -1], [2, -1, -1], [3, -1, -1]],
                          [[12, -1, -1], [13, -1, -1], [14, 10, -1], [11, -1, -1]]],
                         [[[8, -1, -1], [9, 5, -1], [6, -1, -1], [7, -1, -1]],
                          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [19, 15, -1]]]])
    for n in range(1, 3):
        agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds._max_n_row_index(n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data,
                              c.points(ddf, 'x', 'y', ds.by('cat2', ds._max_row_index())).data)


def test_categorical_first(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[0, -1, nan, -3],
                          [12, -13, 10, -11]],
                         [[8, -5, 6, -7],
                          [16, -17, 18, -15]]])
    for n in range(1, 3):
        agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.first("plusminus")))
        assert_eq_ndarray(agg.data, solution)


def test_categorical_last(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[4, -1, nan, -3],
                          [12, -13, 14, -11]],
                         [[8, -9, 6, -7],
                          [16, -17, 18, -19]]])
    for n in range(1, 3):
        agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.last("plusminus")))
        assert_eq_ndarray(agg.data, solution)


def test_categorical_first_n(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[[0, 4, nan], [-1, nan, nan], [nan, nan, nan], [-3, nan, nan]],
                          [[12, nan, nan], [-13, nan, nan], [10, 14, nan], [-11, nan, nan]]],
                         [[[8, nan, nan], [-5, -9, nan], [6, nan, nan], [-7, nan, nan]],
                          [[16, nan, nan], [-17, nan, nan], [18, nan, nan], [-15, -19, nan]]]])
    for n in range(1, 3):
        agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.first_n("plusminus", n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data,
                              c.points(ddf, 'x', 'y', ds.by('cat2', ds.first("plusminus"))).data)


def test_categorical_last_n(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    solution = np.array([[[[4, 0, nan], [-1, nan, nan], [nan, nan, nan], [-3, nan, nan]],
                          [[12, nan, nan], [-13, nan, nan], [14, 10, nan], [-11, nan, nan]]],
                         [[[8, nan, nan], [-9, -5, nan], [6, nan, nan], [-7, nan, nan]],
                          [[16, nan, nan], [-17, nan, nan], [18, nan, nan], [-19, -15, nan]]]])
    for n in range(1, 3):
        agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.last_n("plusminus", n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data,
                              c.points(ddf, 'x', 'y', ds.by('cat2', ds.last("plusminus"))).data)


def test_where_max(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray([[16, 6], [11, 1]], coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.max('i32'), 'reverse')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.max('i64'), 'reverse')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.max('f32'), 'reverse')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.max('f64'), 'reverse')), out)

    # Using row index.
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.max('i32'))), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.max('i64'))), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.max('f64'))), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.max('f32'))), out)


def test_where_min(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray([[20, 10], [15, 5]], coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.min('i32'), 'reverse')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.min('i64'), 'reverse')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.min('f32'), 'reverse')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.min('f64'), 'reverse')), out)

    # Using row index.
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.min('i32'))), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.min('i64'))), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.min('f64'))), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.min('f32'))), out)


def test_where_max_n(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_rowindex = np.array([[[ 4,  0,  1,  3, -1, -1],
                              [14, 12, 10, 11, 13, -1]],
                             [[ 8,  6,  5,  7,  9, -1],
                              [18, 16, 15, 17, 19, -1]]])
    sol_reverse = np.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 7):
        # Using row index.
        agg = c.points(ddf, 'x', 'y', ds.where(ds.max_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(ddf, 'x', 'y', ds.where(ds.max('plusminus'))).data)

        # Using another column
        agg = c.points(ddf, 'x', 'y', ds.where(ds.max_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(ddf, 'x', 'y', ds.where(ds.max('plusminus'),
                                                               'reverse')).data)


def test_where_min_n(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_rowindex = np.array([[[3,  1,  0,  4, -1, -1],
                              [13, 11, 10, 12, 14, -1]],
                             [[ 9,  7,  5,  6,  8, -1],
                              [19, 17, 15, 16, 18, -1]]])
    sol_reverse = np.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 7):
        # Using row index.
        agg = c.points(ddf, 'x', 'y', ds.where(ds.min_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(ddf, 'x', 'y', ds.where(ds.min('plusminus'))).data)

        # Using another column
        agg = c.points(ddf, 'x', 'y', ds.where(ds.min_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(ddf, 'x', 'y', ds.where(ds.min('plusminus'),
                                                               'reverse')).data)


def test_where_first(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    # Note reductions like ds.where(ds.first('i32'), 'reverse') are supported,
    # but the same results can be achieved using the simpler ds.first('reverse')
    out = xr.DataArray([[20, 10], [15, 5]], coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.first('i32'), 'reverse')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.first('i64'), 'reverse')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.first('f32'), 'reverse')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.first('f64'), 'reverse')), out)

    # Using row index.
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.first('i32'))), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.first('i64'))), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.first('f64'))), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.first('f32'))), out)


def test_where_last(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    # Note reductions like ds.where(ds.last('i32'), 'reverse') are supported,
    # but the same results can be achieved using the simpler ds.last('reverse')
    out = xr.DataArray([[16, 6], [11, 1]], coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.last('i32'), 'reverse')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.last('i64'), 'reverse')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.last('f32'), 'reverse')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.last('f64'), 'reverse')), out)

    # Using row index.
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.last('i32'))), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.last('i64'))), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.last('f64'))), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.where(ds.last('f32'))), out)


def test_where_first_n(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_rowindex = np.array([[[ 0,  1,  3,  4, -1, -1],
                              [10, 11, 12, 13, 14, -1]],
                             [[ 5,  6,  7,  8,  9, -1],
                              [15, 16, 17, 18, 19, -1]]])
    sol_reverse = np.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 7):
        # Using row index.
        agg = c.points(ddf, 'x', 'y', ds.where(ds.first_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(ddf, 'x', 'y', ds.where(ds.first('plusminus'))).data)

        # Using another column
        agg = c.points(ddf, 'x', 'y', ds.where(ds.first_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(ddf, 'x', 'y', ds.where(ds.first('plusminus'),
                                                               'reverse')).data)


def test_where_last_n(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_rowindex = np.array([[[ 4,  3,  1,  0, -1, -1],
                              [14, 13, 12, 11, 10, -1]],
                             [[ 9,  8,  7,  6,  5, -1],
                              [19, 18, 17, 16, 15, -1]]])
    sol_reverse = np.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 7):
        # Using row index.
        agg = c.points(ddf, 'x', 'y', ds.where(ds.last_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(ddf, 'x', 'y', ds.where(ds.last('plusminus'))).data)

        # Using another column
        agg = c.points(ddf, 'x', 'y', ds.where(ds.last_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data,
                              c.points(ddf, 'x', 'y', ds.where(ds.last('plusminus'),
                                                               'reverse')).data)

def test_summary_by(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions

    # summary(by)
    agg_summary = c.points(ddf, 'x', 'y', ds.summary(by=ds.by("cat")))
    agg_by = c.points(ddf, 'x', 'y', ds.by("cat"))
    assert_eq_xr(agg_summary["by"], agg_by)

    # summary(by, other_reduction)
    agg_summary = c.points(ddf, 'x', 'y', ds.summary(by=ds.by("cat"), max=ds.max("plusminus")))
    agg_max = c.points(ddf, 'x', 'y', ds.max("plusminus"))
    assert_eq_xr(agg_summary["by"], agg_by)
    assert_eq_xr(agg_summary["max"], agg_max)

    # summary(other_reduction, by)
    agg_summary = c.points(ddf, 'x', 'y', ds.summary(max=ds.max("plusminus"), by=ds.by("cat")))
    assert_eq_xr(agg_summary["by"], agg_by)
    assert_eq_xr(agg_summary["max"], agg_max)

    # summary(by, by)
    agg_summary = c.points(ddf, 'x', 'y', ds.summary(by=ds.by("cat"),
                                                     by_any=ds.by("cat", ds.any())))
    agg_by_any = c.points(ddf, 'x', 'y', ds.by("cat", ds.any()))
    assert_eq_xr(agg_summary["by"], agg_by)
    assert_eq_xr(agg_summary["by_any"], agg_by_any)

    # summary(by("cat1"), by("cat2"))
    agg_summary = c.points(ddf, 'x', 'y', ds.summary(by=ds.by("cat"), by2=ds.by("cat2")))
    agg_by2 = c.points(ddf, 'x', 'y', ds.by("cat2"))
    assert_eq_xr(agg_summary["by"], agg_by)
    assert_eq_xr(agg_summary["by2"], agg_by2)


def test_summary_where_n(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_min_n_rowindex = np.array([[[3,  1,  0,  4, -1],
                                    [13, 11, 10, 12, 14]],
                                   [[ 9,  7,  5,  6,  8],
                                    [19, 17, 15, 16, 18]]])
    sol_max_n_rowindex = np.array([[[ 4,  0,  1,  3, -1],
                                    [14, 12, 10, 11, 13]],
                                   [[ 8,  6,  5,  7,  9],
                                    [18, 16, 15, 17, 19]]])
    sol_max_n_reverse = np.where(np.logical_or(sol_max_n_rowindex < 0, sol_max_n_rowindex == 6),
                                 np.nan, 20 - sol_max_n_rowindex)

    agg = c.points(ddf, 'x', 'y', ds.summary(
        count=ds.count(),
        min_n=ds.where(ds.min_n('plusminus', 5)),
        max_n=ds.where(ds.max_n('plusminus', 5), 'reverse'),
    ))
    assert_eq_ndarray(agg.coords['n'], np.arange(5))

    assert agg['count'].dims == ('y', 'x')
    assert agg['min_n'].dims == ('y', 'x', 'n')
    assert agg['max_n'].dims == ('y', 'x', 'n')

    assert agg['count'].dtype == np.dtype('uint32')
    assert agg['min_n'].dtype == np.dtype('int64')
    assert agg['max_n'].dtype == np.dtype('float64')

    assert_eq_ndarray(agg['count'].data, [[5, 5], [5, 5]])
    assert_eq_ndarray(agg['min_n'].data, sol_min_n_rowindex)
    assert_eq_ndarray(agg['max_n'].data, sol_max_n_reverse)


def test_mean(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray(
        values(_pandas().i32).reshape((2, 2, 5)).mean(axis=2, dtype='f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.mean('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.mean('i64')), out)
    out = xr.DataArray(
        np.nanmean(values(_pandas().f64).reshape((2, 2, 5)), axis=2).T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.mean('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.mean('f64')), out)


def test_var(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray(
        values(_pandas().i32).reshape((2, 2, 5)).var(axis=2, dtype='f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.var('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.var('i64')), out)
    out = xr.DataArray(
        np.nanvar(values(_pandas().f64).reshape((2, 2, 5)), axis=2).T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.var('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.var('f64')), out)


def test_std(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    out = xr.DataArray(
        values(_pandas().i32).reshape((2, 2, 5)).std(axis=2, dtype='f8').T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.std('i32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.std('i64')), out)
    out = xr.DataArray(
        np.nanstd(values(_pandas().f64).reshape((2, 2, 5)), axis=2).T,
        coords=coords, dims=dims)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.std('f32')), out)
    assert_eq_xr(c.points(ddf, 'x', 'y', ds.std('f64')), out)


def test_count_cat(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol = np.array([[[5, 0, 0, 0],
                     [0, 0, 5, 0]],
                    [[0, 5, 0, 0],
                     [0, 0, 0, 5]]])
    out = xr.DataArray(
        sol, coords=(coords + [['a', 'b', 'c', 'd']]), dims=(dims + ['cat'])
    )
    agg = c.points(ddf, 'x', 'y', ds.count_cat('cat'))
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 1), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)

    # categorizing by (cat_int-10)%4 ought to give the same result
    out = xr.DataArray(
        sol, coords=(coords + [range(4)]), dims=(dims + ['cat_int'])
    )
    agg = c.points(ddf, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.count()))
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 1), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)

    # easier to write these tests in here, since we expect the same result with only slight tweaks

    # add an extra category (this will count nans and out of bounds)
    sol = np.append(sol, [[[0], [0]],[[0], [0]]], axis=2)

    # categorizing by binning the integer arange columns using [0,20] into 4 bins. Same result as
    # for count_cat
    for col in 'i32', 'i64':
        out = xr.DataArray(
            sol, coords=(coords + [range(5)]), dims=(dims + [col])
        )
        agg = c.points(ddf, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.count()))
        assert_eq_xr(agg, out)
        assert_eq_ndarray(agg.x_range, (0, 1), close=True)
        assert_eq_ndarray(agg.y_range, (0, 1), close=True)

    # as above, but for the float arange columns. Element 2 has a nan, so the first bin is one
    # short, and the nan bin is +1
    sol[0, 0, 0] = 4
    sol[0, 0, 4] = 1

    for col in 'f32', 'f64':
        out = xr.DataArray(
            sol, coords=(coords + [range(5)]), dims=(dims + [col])
        )
        agg = c.points(ddf, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.count()))
        assert_eq_xr(agg, out)
        assert_eq_ndarray(agg.x_range, (0, 1), close=True)
        assert_eq_ndarray(agg.y_range, (0, 1), close=True)


def test_categorical_sum(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol = np.array([[[ 10, nan, nan, nan],
                     [nan, nan,  60, nan]],
                    [[nan,  35, nan, nan],
                     [nan, nan, nan,  85]]])
    out = xr.DataArray(
        sol, coords=(coords + [['a', 'b', 'c', 'd']]), dims=(dims + ['cat'])
    )
    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.sum('i32')))
    assert_eq_xr(agg, out)

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.sum('i64')))
    assert_eq_xr(agg, out)

    out = xr.DataArray(
        sol, coords=(coords + [range(4)]), dims=(dims + ['cat_int'])
    )
    agg = c.points(ddf, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.sum('i32')))
    assert_eq_xr(agg, out)

    agg = c.points(ddf, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.sum('i64')))
    assert_eq_xr(agg, out)

    sol = np.array([[[8.0,  nan,  nan,  nan],
                     [nan,  nan, 60.0,  nan]],
                    [[nan, 35.0,  nan,  nan],
                     [nan,  nan,  nan, 85.0]]])
    out = xr.DataArray(
        sol, coords=(coords + [['a', 'b', 'c', 'd']]), dims=(dims + ['cat'])
    )
    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.sum('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.sum('f64')))
    assert_eq_xr(agg, out)


def test_categorical_sum_binning(ddf, npartitions, request):
    if "cudf" in request.node.name:
        pytest.skip(
            "The categorical binning of 'sum' reduction is yet supported on the GPU"
        )
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol = np.array([[[8.0,  nan,  nan,  nan],
                     [nan,  nan, 60.0,  nan]],
                    [[nan, 35.0,  nan,  nan],
                     [nan,  nan,  nan, 85.0]]])

    # add an extra category (this will count nans and out of bounds)
    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(
            sol, coords=(coords + [range(5)]), dims=(dims + [col])
        )
        agg = c.points(ddf, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.sum(col)))
        assert_eq_xr(agg, out)
        assert_eq_ndarray(agg.x_range, (0, 1), close=True)
        assert_eq_ndarray(agg.y_range, (0, 1), close=True)


def test_categorical_mean(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol = np.array([[[  2, nan, nan, nan],
                     [nan, nan,  12, nan]],
                    [[nan,   7, nan, nan],
                     [nan, nan, nan,  17]]])
    out = xr.DataArray(
        sol,
        coords=(coords + [['a', 'b', 'c', 'd']]),
        dims=(dims + ['cat']))

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.mean('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.mean('f64')))
    assert_eq_xr(agg, out)

    out = xr.DataArray(
        sol, coords=(coords + [range(4)]), dims=(dims + ['cat_int'])
    )
    agg = c.points(ddf, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.mean('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(ddf, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.mean('f64')))
    assert_eq_xr(agg, out)


def test_categorical_mean_binning(ddf, npartitions, request):
    if "cudf" in request.node.name:
        pytest.skip(
            "The categorical binning of 'mean' reduction is yet supported on the GPU"
        )
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol = np.array([[[  2, nan, nan, nan],
                     [nan, nan,  12, nan]],
                    [[nan,   7, nan, nan],
                     [nan, nan, nan,  17]]])

    # add an extra category (this will count nans and out of bounds)
    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(
            sol, coords=(coords + [range(5)]), dims=(dims + [col])
        )
        agg = c.points(ddf, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.mean(col)))
        assert_eq_xr(agg, out)


def test_categorical_var(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol = np.array([[[ 2.5,  nan,  nan,  nan],
                     [ nan,  nan,   2.,  nan]],
                    [[ nan,   2.,  nan,  nan],
                     [ nan,  nan,  nan,   2.]]])
    out = xr.DataArray(
        sol,
        coords=(coords + [['a', 'b', 'c', 'd']]),
        dims=(dims + ['cat']))

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.var('f32')))
    assert_eq_xr(agg, out, True)

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.var('f64')))
    assert_eq_xr(agg, out, True)

    out = xr.DataArray(
        sol, coords=(coords + [range(4)]), dims=(dims + ['cat_int'])
    )
    agg = c.points(ddf, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.var('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(ddf, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.var('f64')))
    assert_eq_xr(agg, out)

    # add an extra category (this will count nans and out of bounds)
    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(
            sol, coords=(coords + [range(5)]), dims=(dims + [col])
        )
        agg = c.points(ddf, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.var(col)))
        assert_eq_xr(agg, out)


def test_categorical_std(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol = np.sqrt(np.array([
        [[ 2.5,  nan,  nan,  nan],
         [ nan,  nan,   2.,  nan]],
        [[ nan,   2.,  nan,  nan],
         [ nan,  nan,  nan,   2.]]])
    )
    out = xr.DataArray(
        sol,
        coords=(coords + [['a', 'b', 'c', 'd']]),
        dims=(dims + ['cat']))

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.std('f32')))
    assert_eq_xr(agg, out, True)

    agg = c.points(ddf, 'x', 'y', ds.by('cat', ds.std('f64')))
    assert_eq_xr(agg, out, True)

    out = xr.DataArray(
        sol, coords=(coords + [range(4)]), dims=(dims + ['cat_int'])
    )
    agg = c.points(ddf, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.std('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(ddf, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10),
                                        ds.std('f64')))
    assert_eq_xr(agg, out)

    # add an extra category (this will count nans and out of bounds)
    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(
            sol, coords=(coords + [range(5)]), dims=(dims + [col])
        )
        agg = c.points(ddf, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.std(col)))
        assert_eq_xr(agg, out)


def test_multiple_aggregates(ddf, npartitions):
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    agg = c.points(ddf, 'x', 'y',
                   ds.summary(f64_std=ds.std('f64'),
                              f64_mean=ds.mean('f64'),
                              i32_sum=ds.sum('i32'),
                              i32_count=ds.count('i32')))

    def f(x):
        return xr.DataArray(x, coords=coords, dims=dims)
    assert_eq_xr(agg.f64_std, f(np.nanstd(values(_pandas().f64).reshape((2, 2, 5)), axis=2).T))
    assert_eq_xr(agg.f64_mean, f(np.nanmean(values(_pandas().f64).reshape((2, 2, 5)), axis=2).T))
    assert_eq_xr(agg.i32_sum, f(values(_pandas().i32).reshape((2, 2, 5)).sum(axis=2, dtype='f8').T))
    assert_eq_xr(agg.i32_count, f(np.array([[5, 5], [5, 5]], dtype='i4')))

    assert_eq_ndarray(agg.x_range, (0, 1), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)


def test_auto_range_points(DataFrame):
    n = 10
    data = np.arange(n, dtype='i4')

    ddf = DataFrame({'time': np.arange(n),
                     'x': data,
                     'y': data})

    cvs = ds.Canvas(plot_width=n, plot_height=n)
    agg = cvs.points(ddf, 'x', 'y', ds.count('time'))
    sol = np.zeros((n, n), int)
    np.fill_diagonal(sol, 1)
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 9), close=True)
    assert_eq_ndarray(agg.y_range, (0, 9), close=True)

    cvs = ds.Canvas(plot_width=n+1, plot_height=n+1)
    agg = cvs.points(ddf, 'x', 'y', ds.count('time'))
    sol = np.zeros((n+1, n+1), int)
    np.fill_diagonal(sol, 1)
    sol[5, 5] = 0
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 9), close=True)
    assert_eq_ndarray(agg.y_range, (0, 9), close=True)

    n = 4
    data = np.arange(n, dtype='i4')
    ddf = DataFrame({'time': np.arange(n),
                     'x': data,
                     'y': data})

    cvs = ds.Canvas(plot_width=2*n, plot_height=2*n)
    agg = cvs.points(ddf, 'x', 'y', ds.count('time'))
    sol = np.zeros((2*n, 2*n), int)
    np.fill_diagonal(sol, 1)
    sol[np.array([tuple(range(1, 4, 2))])] = 0
    sol[np.array([tuple(range(4, 8, 2))])] = 0
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 3), close=True)
    assert_eq_ndarray(agg.y_range, (0, 3), close=True)

    cvs = ds.Canvas(plot_width=2*n+1, plot_height=2*n+1)
    agg = cvs.points(ddf, 'x', 'y', ds.count('time'))
    sol = np.zeros((2*n+1, 2*n+1), int)
    sol[0, 0] = 1
    sol[3, 3] = 1
    sol[6, 6] = 1
    sol[8, 8] = 1
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 3), close=True)
    assert_eq_ndarray(agg.y_range, (0, 3), close=True)


def test_uniform_points(DataFrame):
    n = 101
    ddf = DataFrame({'time': np.ones(2*n, dtype='i4'),
                     'x': np.concatenate((np.arange(n, dtype='f8'),
                                          np.arange(n, dtype='f8'))),
                     'y': np.concatenate(([0.] * n, [1.] * n))})

    cvs = ds.Canvas(plot_width=10, plot_height=2, y_range=(0, 1))
    agg = cvs.points(ddf, 'x', 'y', ds.count('time'))
    sol = np.array([[10] * 9 + [11], [10] * 9 + [11]], dtype='i4')
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 100), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)


@pytest.mark.parametrize('high', [9, 10, 99, 100])
@pytest.mark.parametrize('low', [0])
def test_uniform_diagonal_points(DataFrame, low, high):
    bounds = (low, high)
    x_range, y_range = bounds, bounds

    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    n = width * height
    ddf = DataFrame({'time': np.ones(n, dtype='i4'),
                     'x': np.array([np.arange(*x_range, dtype='f8')] * width).flatten(),
                     'y': np.array([np.arange(*y_range, dtype='f8')] * height).flatten()})

    cvs = ds.Canvas(plot_width=2, plot_height=2, x_range=x_range, y_range=y_range)
    agg = cvs.points(ddf, 'x', 'y', ds.count('time'))

    diagonal = agg.data.diagonal(0)
    assert sum(diagonal) == n
    assert abs(bounds[1] - bounds[0]) % 2 == abs(diagonal[1] / high - diagonal[0] / high)

    assert_eq_ndarray(agg.x_range, (low, high), close=True)
    assert_eq_ndarray(agg.y_range, (low, high), close=True)


def test_log_axis_points(ddf):
    axis = ds.core.LogAxis()
    logcoords = axis.compute_index(axis.compute_scale_and_translate((1, 10), 2), 2)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)

    sol = np.array([[5, 5], [5, 5]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, logcoords],
                       dims=['y', 'log_x'])
    assert_eq_xr(c_logx.points(ddf, 'log_x', 'y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, lincoords],
                       dims=['log_y', 'x'])
    assert_eq_xr(c_logy.points(ddf, 'x', 'log_y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, logcoords],
                       dims=['log_y', 'log_x'])
    assert_eq_xr(c_logxy.points(ddf, 'log_x', 'log_y', ds.count('i32')), out)


@pytest.mark.skipif(not sp, reason="spatialpandas not installed")
@dask_switcher(query=False, extras=["spatialpandas.dask"])
def test_points_geometry():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0., 2.), 3), 3)

    ddf = dd.from_pandas(sp.GeoDataFrame({
        'geom': pd.array(
            [[0, 0], [0, 1, 1, 1], [0, 2, 1, 2, 2, 2]], dtype='MultiPoint[float64]'),
        'v': [1, 2, 3]
    }), npartitions=3)

    cvs = ds.Canvas(plot_width=3, plot_height=3)
    agg = cvs.points(ddf, geometry='geom', agg=ds.sum('v'))
    sol = np.array([[1, nan, nan],
                    [2, 2,   nan],
                    [3, 3,   3]], dtype='float64')
    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


@dask_switcher(query=False, extras=["spatialpandas.dask"])
def test_line(DataFrame):
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-3., 3.), 7), 7)

    ddf = DataFrame({'x': [4, 0, -4, -3, -2, -1.9, 0, 10, 10, 0, 4],
                     'y': [0, -4, 0, 1, 2, 2.1, 4, 20, 30, 4, 0]})
    cvs = ds.Canvas(plot_width=7, plot_height=7,
                    x_range=(-3, 3), y_range=(-3, 3))
    agg = cvs.line(ddf, 'x', 'y', ds.count())
    sol = np.array([[0, 0, 1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1],
                    [0, 2, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (-3, 3), close=True)
    assert_eq_ndarray(agg.y_range, (-3, 3), close=True)


# # Line tests
line_manual_range_params = [
    # axis1 none constant
    (dict(data={
        'x0': [4, -4, 4],
        'x1': [0,  0, 0],
        'x2': [-4, 4, -4],
        'y0': [0,  0, 0],
        'y1': [-4, 4, 0],
        'y2': [0,  0, 0]
    }), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis1 x constant
    (dict(data={
        'y0': [0, 0,  0],
        'y1': [0, 4, -4],
        'y2': [0, 0,  0]
    }), dict(x=np.array([-4, 0, 4]), y=['y0', 'y1', 'y2'], axis=1)),

    # axis0 single
    (dict(data={
        'x': [4, 0, -4, np.nan, -4, 0, 4, np.nan, 4, 0, -4],
        'y': [0, -4, 0, np.nan, 0, 4, 0, np.nan, 0, 0, 0],
    }), dict(x='x', y='y', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [4,  0, -4],
        'x1': [-4, 0,  4],
        'x2': [4,  0, -4],
        'y0': [0, -4,  0],
        'y1': [0,  4,  0],
        'y2': [0,  0,  0]
    }), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=0)),

    # axis0 multi with string
    (dict(data={
        'x0': [-4,  0, 4],
        'y0': [0, -4,  0],
        'y1': [0,  4,  0],
        'y2': [0,  0,  0]
    }), dict(x='x0', y=['y0', 'y1', 'y2'], axis=0)),

    # axis1 RaggedArray
    (dict(data={
        'x': RaggedArray([[4, 0, -4], [-4, 0, 4, 4, 0, -4]]),
        'y': RaggedArray([[0, -4, 0], [0, 4, 0, 0, 0, 0]]),
    }, dtype='Ragged[int64]'), dict(x='x', y='y', axis=1)),
]
if sp:
    line_manual_range_params.append(
        # geometry
        (dict(data={
            'geom': [[4, 0, 0, -4, -4, 0],
                     [-4, 0, 0, 4, 4, 0, 4, 0, 0, 0, -4, 0]]
        }, dtype='Line[int64]'), dict(geometry='geom'))
    )

@dask_switcher(query=False, extras=["spatialpandas.dask"])
@pytest.mark.parametrize('df_kwargs,cvs_kwargs', line_manual_range_params[5:7])
def test_line_manual_range(DataFrame, df_kwargs, cvs_kwargs, request):
    if "cudf" in request.node.name:
        dtype = df_kwargs.get('dtype', '')
        if dtype.startswith('Ragged') or dtype.startswith('Line'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-3., 3.), 7), 7)

    ddf = DataFrame(geo='geometry' in cvs_kwargs, **df_kwargs)
    cvs = ds.Canvas(plot_width=7, plot_height=7,
                    x_range=(-3, 3), y_range=(-3, 3))

    agg = cvs.line(ddf, agg=ds.count(), **cvs_kwargs)

    if (ddf.npartitions == 2 and cvs_kwargs.get('axis') == 0 and
            isinstance(cvs_kwargs['y'], (list, tuple))):
        # Github issue #1106.
        # When axis==0 we do not deal with dask splitting up our lines/areas,
        # so the output has undesirable missing segments.
        if isinstance(cvs_kwargs['x'], list):
            sol = np.array([[0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0]], dtype='i4')
        else:
            sol = np.array([[0, 0, 1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0]], dtype='i4')
    else:
        # Ideally all tests would give this solution.
        sol = np.array([[0, 0, 1, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 1, 0, 0]], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (-3, 3), close=True)
    assert_eq_ndarray(agg.y_range, (-3, 3), close=True)


line_autorange_params = [
    # axis1 none constant
    (dict(data={
        'x0': [0,  0, 0],
        'x1': [-4, 0, 4],
        'x2': [0,  0, 0],
        'y0': [-4, 4, -4],
        'y1': [0,  0,  0],
        'y2': [4, -4,  4]
    }), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis1 y constant
    (dict(data={
        'x0': [0,  0, 0],
        'x1': [-4, 0, 4],
        'x2': [0,  0, 0],
    }), dict(x=['x0', 'x1', 'x2'], y=np.array([-4, 0, 4]), axis=1)),

    # axis0 single
    (dict(data={
        'x': [0, -4, 0, np.nan, 0, 0, 0, np.nan, 0, 4, 0],
        'y': [-4, 0, 4, np.nan, 4, 0, -4, np.nan, -4, 0, 4],
    }), dict(x='x', y='y', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [0, -4,  0],
        'x1': [0,  0,  0],
        'x2': [0,  4,  0],
        'y0': [-4, 0,  4],
        'y1': [4,  0, -4],
        'y2': [-4, 0,  4]
    }), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=0)),

    # axis0 multi with string
    (dict(data={
        'x0': [0, -4,  0],
        'x1': [0,  0,  0],
        'x2': [0,  4,  0],
        'y0': [-4, 0,  4]
    }), dict(x=['x0', 'x1', 'x2'], y='y0', axis=0)),

    # axis1 RaggedArray
    (dict(data={
        'x': [[0, -4, 0], [0, 0, 0], [0, 4, 0]],
        'y': [[-4, 0, 4], [4, 0, -4], [-4, 0, 4]],
    }, dtype='Ragged[int64]'), dict(x='x', y='y', axis=1)),
]
if sp:
    line_autorange_params.append(
        # geometry
        (dict(data={
            'geom': [[0, -4, -4, 0, 0, 4],
                     [0, 4, 0, 0, 0, -4],
                     [0, -4, 4, 0, 0, 4]]
        }, dtype='Line[int64]'), dict(geometry='geom'))
    )

@dask_switcher(query=False, extras=["spatialpandas.dask"])
@pytest.mark.parametrize('df_kwargs,cvs_kwargs', line_autorange_params)
def test_line_autorange(DataFrame, df_kwargs, cvs_kwargs, request):
    if "cudf" in request.node.name:
        dtype = df_kwargs.get('dtype', '')
        if dtype.startswith('Ragged') or dtype.startswith('Line'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 9), 9)

    ddf = DataFrame(geo='geometry' in cvs_kwargs, **df_kwargs)

    cvs = ds.Canvas(plot_width=9, plot_height=9)

    agg = cvs.line(ddf, agg=ds.count(), **cvs_kwargs)

    if (ddf.npartitions == 2 and cvs_kwargs.get('axis') == 0 and
            isinstance(cvs_kwargs['x'], (list, tuple))):
        # Github issue #1106.
        # When axis==0 we do not deal with dask splitting up our lines/areas,
        # so the output has undesirable missing segments.
        if isinstance(cvs_kwargs['y'], list):
            sol = np.array([[0, 0, 0, 0, 2, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 1, 0],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='i4')
        else:
            sol = np.array([[0, 0, 0, 0, 3, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 1, 0, 1, 0, 0],
                            [0, 1, 0, 0, 1, 0, 0, 1, 0],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='i4')
    else:
        sol = np.array([[0, 0, 0, 0, 3, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 1, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 1, 0],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 1, 0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 1, 0, 1, 0, 1, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 3, 0, 0, 0, 0]], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (-4, 4), close=True)
    assert_eq_ndarray(agg.y_range, (-4, 4), close=True)


def test_line_x_constant_autorange(DataFrame):
    # axis1 y constant
    x = np.array([-4, 0, 4])
    y = ['y0', 'y1', 'y2']
    ax = 1

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 9), 9)

    ddf = DataFrame({
        'y0': [0, 0, 0],
        'y1': [-4, 0, 4],
        'y2': [0, 0, 0],
    })

    cvs = ds.Canvas(plot_width=9, plot_height=9)

    agg = cvs.line(ddf, x, y, ds.count(), axis=ax)

    sol = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [3, 1, 1, 1, 1, 1, 1, 1, 3],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


def test_log_axis_line(ddf):
    axis = ds.core.LogAxis()
    logcoords = axis.compute_index(axis.compute_scale_and_translate((1, 10), 2), 2)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)

    sol = np.array([[4, 5], [5, 5]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, logcoords],
                       dims=['y', 'log_x'])

    assert_eq_xr(c_logx.line(ddf, 'log_x', 'y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, lincoords],
                       dims=['log_y', 'x'])
    assert_eq_xr(c_logy.line(ddf, 'x', 'log_y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, logcoords],
                       dims=['log_y', 'log_x'])
    assert_eq_xr(c_logxy.line(ddf, 'log_x', 'log_y', ds.count('i32')), out)


def test_auto_range_line(DataFrame):
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-10., 10.), 5), 5)

    ddf = DataFrame({'x': [-10,  0, 10,   0, -10],
                     'y': [  0, 10,  0, -10,   0]})

    cvs = ds.Canvas(plot_width=5, plot_height=5)
    agg = cvs.line(ddf, 'x', 'y', ds.count())
    sol = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [2, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (-10, 10), close=True)
    assert_eq_ndarray(agg.y_range, (-10, 10), close=True)


@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [
    # axis1 none constant
    (dict(data={
        'x0': [-4, np.nan],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, np.nan],
        'y1': [-4, 4],
        'y2': [0, 0]
    }, dtype='float32'), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis0 single
    (dict(data={
        'x': [-4, -2, 0, np.nan, 2, 4],
        'y': [0, -4, 0, np.nan, 4, 0],
    }), dict(x='x', y='y', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [np.nan, 2, 4],
        'y0': [0, -4, 0],
        'y1': [np.nan, 4, 0],
    }, dtype='float32'),  dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': pd.array([[-4, -2, 0], [2, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4, 0], [4, 0]], dtype='Ragged[float32]')
    }, dtype='Ragged[float32]'), dict(x='x', y='y', axis=1))
])
def test_area_to_zero_fixedrange(DataFrame, df_kwargs, cvs_kwargs):
    if DataFrame in (_dask_cudf_DataFrame, _dask_expr_cudf_DataFrame):
        if df_kwargs.get('dtype', '').startswith('Ragged'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-2.25, 2.25), 5), 5)

    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-3.75, 3.75), 9), 9)

    cvs = ds.Canvas(plot_width=9, plot_height=5,
                    x_range=[-3.75, 3.75], y_range=[-2.25, 2.25])

    ddf = DataFrame(**df_kwargs)

    agg = cvs.area(ddf, agg=ds.count(), **cvs_kwargs)

    if (ddf.npartitions == 2 and cvs_kwargs.get('axis') == 0 and
            isinstance(cvs_kwargs['x'], (list, tuple))):
        # Github issue #1106.
        # When axis==0 we do not deal with dask splitting up our lines/areas,
        # so the output has undesirable missing segments.
        sol = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                       dtype='i4')
    else:
        sol = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 1, 0]],
                       dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (-3.75, 3.75), close=True)
    assert_eq_ndarray(agg.y_range, (-2.25, 2.25), close=True)


@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [
    # axis1 none constant
    (dict(data={
        'x0': [-4, 0],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, 0],
        'y1': [-4, -4],
        'y2': [0, 0]
    }, dtype='float32'),
     dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis1 y constant
    (dict(data={
        'x0': [-4, 0],
        'x1': [-2, 2],
        'x2': [0, 4],
    }, dtype='float32'),
     dict(x=['x0', 'x1', 'x2'], y=np.array([0, -4, 0], dtype='float32'), axis=1)),

    # axis0 single
    (dict(data={
        'x': [-4, -2, 0, 0, 2, 4],
        'y': [0, -4, 0, 0, -4, 0],
    }), dict(x='x', y='y', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [0, 2, 4],
        'y0': [0, -4, 0],
        'y1': [0, -4, 0],
    }, dtype='float32'), dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)),

    # axis0 multi, y string
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [0, 2, 4],
        'y0': [0, -4, 0],
    }, dtype='float32'), dict(x=['x0', 'x1'], y='y0', axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': [[-4, -2, 0], [0, 2, 4]],
        'y': [[0, -4, 0], [0, -4, 0]]
    }, dtype='Ragged[float32]'), dict(x='x', y='y', axis=1))
])
def test_area_to_zero_autorange(DataFrame, df_kwargs, cvs_kwargs):
    if DataFrame in (_dask_cudf_DataFrame, _dask_expr_cudf_DataFrame):
        if df_kwargs.get('dtype', '').startswith('Ragged'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 0.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    ddf = DataFrame(**df_kwargs)
    agg = cvs.area(ddf, agg=ds.count(), **cvs_kwargs)

    if (ddf.npartitions == 2 and cvs_kwargs.get('axis') == 0 and
            isinstance(cvs_kwargs['x'], (list, tuple))):
        # Github issue #1106.
        # When axis==0 we do not deal with dask splitting up our lines/areas,
        # so the output has undesirable missing segments.
        sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]],
                       dtype='i4')
    else:
        sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]],
                       dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (-4, 4), close=True)
    assert_eq_ndarray(agg.y_range, (-4, 0), close=True)


@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [
    # axis1 none constant
    (dict(data={
        'x0': [-4, np.nan],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, np.nan],
        'y1': [-4, 4],
        'y2': [0, 0]
    }, dtype='float32'), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis0 single
    (dict(data={
        'x': [-4, -2, 0, np.nan, 2, 4],
        'y': [0, -4, 0, np.nan, 4, 0],
    }), dict(x='x', y='y', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [np.nan, 2, 4],
        'y0': [0, -4, 0],
        'y1': [np.nan, 4, 0],
    }, dtype='float32'), dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': [[-4, -2, 0], [2, 4]],
        'y': [[0, -4, 0], [4, 0]],
    }, dtype='Ragged[float32]'), dict(x='x', y='y', axis=1))
])
def test_area_to_zero_autorange_gap(DataFrame, df_kwargs, cvs_kwargs):
    if DataFrame in (_dask_cudf_DataFrame, _dask_expr_cudf_DataFrame):
        if df_kwargs.get('dtype', '').startswith('Ragged'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    ddf = DataFrame(**df_kwargs)

    agg = cvs.area(ddf, agg=ds.count(), **cvs_kwargs)

    if (ddf.npartitions == 2 and cvs_kwargs.get('axis') == 0 and
            isinstance(cvs_kwargs['x'], (list, tuple))):
        # Github issue #1106.
        # When axis==0 we do not deal with dask splitting up our lines/areas,
        # so the output has undesirable missing segments.
        sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                       dtype='i4')
    else:
        sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
                       dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [
    # axis1 none constant
    (dict(data={
        'x0': [-4, 0],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, 0],
        'y1': [-4, -4],
        'y2': [0, 0],
        'y3': [0, 0],
        'y4': [-2, -2],
        'y5': [0, 0],
    }, dtype='float32'),
     dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'],
          y_stack=['y3', 'y4', 'y5'], axis=1)),

    # axis1 y constant
    (dict(data={
        'x0': [-4, 0],
        'x1': [-2, 2],
        'x2': [0, 4],
    }, dtype='float32'),
     dict(x=['x0', 'x1', 'x2'], y=np.array([0, -4, 0]),
          y_stack=np.array([0, -2, 0], dtype='float32'), axis=1)),

    # axis0 single
    (dict(data={
        'x': [-4, -2, 0, 0, 2, 4],
        'y': [0, -4, 0, 0, -4, 0],
        'y_stack': [0, -2, 0, 0, -2, 0],
    }), dict(x='x', y='y', y_stack='y_stack', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [0, 2, 4],
        'y0': [0, -4, 0],
        'y1': [0, -4, 0],
        'y2': [0, -2, 0],
        'y3': [0, -2, 0],
    }, dtype='float32'),
     dict(x=['x0', 'x1'], y=['y0', 'y1'], y_stack=['y2', 'y3'], axis=0)),

    # axis0 multi, y string
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [0, 2, 4],
        'y0': [0, -4, 0],
        'y2': [0, -2, 0],
    }, dtype='float32'), dict(x=['x0', 'x1'], y='y0', y_stack='y2', axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': [[-4, -2, 0], [0, 2, 4]],
        'y': [[0, -4, 0], [0, -4, 0]],
        'y_stack': [[0, -2, 0], [0, -2, 0]]
    }, dtype='Ragged[float32]'), dict(x='x', y='y', y_stack='y_stack', axis=1))
])
def test_area_to_line_autorange(DataFrame, df_kwargs, cvs_kwargs):
    if DataFrame in (_dask_cudf_DataFrame, _dask_expr_cudf_DataFrame):
        if df_kwargs.get('dtype', '').startswith('Ragged'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 0.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    ddf = DataFrame(**df_kwargs)
    agg = cvs.area(ddf, agg=ds.count(), **cvs_kwargs)

    if (ddf.npartitions == 2 and cvs_kwargs.get('axis') == 0 and
            isinstance(cvs_kwargs['x'], (list, tuple))):
        # Github issue #1106.
        # When axis==0 we do not deal with dask splitting up our lines/areas,
        # so the output has undesirable missing segments.
        sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                       dtype='i4')
    else:
        sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                       dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (-4, 4), close=True)
    assert_eq_ndarray(agg.y_range, (-4, 0), close=True)


@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [
    # axis1 none constant
    (dict(data={
        'x0': [-4, np.nan],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, np.nan],
        'y1': [-4, 4],
        'y2': [0, 0],
        'y4': [0, 0],
        'y5': [0, 0],
        'y6': [0, 0]
    }, dtype='float32'),
     dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'],
          y_stack=['y4', 'y5', 'y6'], axis=1)),

    # axis0 single
    (dict(data={
        'x': [-4, -2, 0, np.nan, 2, 4],
        'y': [0, -4, 0, np.nan, 4, 0],
        'y_stack': [0, 0, 0, 0, 0, 0],
    }), dict(x='x', y='y', y_stack='y_stack', axis=0)),

    # axis0 multi
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [np.nan, 2, 4],
        'y0': [0, -4, 0],
        'y1': [np.nan, 4, 0],
        'y2': [0, 0, 0],
        'y3': [0, 0, 0],
    }, dtype='float32'),
     dict(x=['x0', 'x1'], y=['y0', 'y1'], y_stack=['y2', 'y3'], axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': [[-4, -2, 0], [2, 4]],
        'y': [[0, -4, 0], [4, 0]],
        'y_stack': [[0, 0, 0], [0, 0]],
    }, dtype='Ragged[float32]'), dict(x='x', y='y', y_stack='y_stack', axis=1))
])
def test_area_to_line_autorange_gap(DataFrame, df_kwargs, cvs_kwargs):
    if DataFrame in (_dask_cudf_DataFrame, _dask_expr_cudf_DataFrame):
        if df_kwargs.get('dtype', '').startswith('Ragged'):
            pytest.skip("Ragged array not supported with cudf")

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    ddf = DataFrame(**df_kwargs)

    # When a line is specified to fill to, this line is not included in
    # the fill.  So we expect the y=0 line to not be filled.
    agg = cvs.area(ddf, agg=ds.count(), **cvs_kwargs)

    if (ddf.npartitions == 2 and cvs_kwargs.get('axis') == 0 and
            isinstance(cvs_kwargs['x'], (list, tuple))):
        # Github issue #1106.
        # When axis==0 we do not deal with dask splitting up our lines/areas,
        # so the output has undesirable missing segments.
        sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                       dtype='i4')
    else:
        sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
                       dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


def test_trimesh_no_double_edge():
    """Assert that when two triangles share an edge that would normally get
    double-drawn, the edge is only drawn for the rightmost (or bottommost)
    triangle.
    """
    import multiprocessing as mp
    # Test left/right edge shared
    verts = dd.from_pandas(pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                                         'y': [4, 5, 5, 5, 4, 4]}), npartitions=mp.cpu_count())
    tris = dd.from_pandas(pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [1, 2]}),
                          npartitions=mp.cpu_count())
    # Plot dims and x/y ranges need to be set such that the edge is drawn twice:
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)
    assert_eq_ndarray(agg.x_range, (0, 5), close=True)
    assert_eq_ndarray(agg.y_range, (0, 5), close=True)


@pytest.mark.parametrize('npartitions', list(range(1, 6)))
def test_trimesh_dask_partitions(npartitions):
    """Assert that when two triangles share an edge that would normally get
    double-drawn, the edge is only drawn for the rightmost (or bottommost)
    triangle.
    """
    # Test left/right edge shared
    verts = dd.from_pandas(pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                                         'y': [4, 5, 5, 5, 4, 4]}),
                           npartitions=npartitions)
    tris = dd.from_pandas(
        pd.DataFrame(
            {'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [1, 2]}),
        npartitions=npartitions)

    cvs = ds.Canvas(plot_width=20, plot_height=20,
                    x_range=(0, 5), y_range=(0, 5))

    # Precompute mesh with dask dataframes
    mesh = du.mesh(verts, tris)

    # Make sure mesh is a dask DataFrame
    assert isinstance(mesh, dd.DataFrame)

    # Check mesh length
    n = len(mesh)
    assert n == 6

    # Make sure we have expected number of partitions
    expected_chunksize = int(np.ceil(len(mesh) / (3*npartitions)) * 3)
    expected_npartitions = int(np.ceil(n / expected_chunksize))
    assert expected_npartitions == mesh.npartitions

    # Make sure triangles don't straddle partitions
    partitions_lens = mesh.map_partitions(len).compute()
    for partitions_len in partitions_lens:
        assert partitions_len % 3 == 0

    agg = cvs.trimesh(verts, tris, mesh)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(
        np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)
    assert_eq_ndarray(agg.x_range, (0, 5), close=True)
    assert_eq_ndarray(agg.y_range, (0, 5), close=True)


@pytest.mark.parametrize('reduction,dtype,aa_dtype', [
    (ds.any(), bool, np.float32),
    (ds.count(), np.uint32, np.float32),
    (ds.max("f64"), np.float64, np.float64),
    (ds.min("f64"), np.float64, np.float64),
    (ds.sum("f64"), np.float64, np.float64),
])
def test_combine_dtype(ddf, reduction, dtype, aa_dtype, request):
    if "cudf" in request.node.name:
        pytest.skip("antialiased lines not supported with cudf")

    cvs = ds.Canvas(plot_width=10, plot_height=10)

    # Non-antialiased lines
    agg = cvs.line(ddf, 'x', 'y', line_width=0, agg=reduction)
    assert agg.dtype == dtype

    # Antialiased lines
    agg = cvs.line(ddf, 'x', 'y', line_width=1, agg=reduction)
    assert agg.dtype == aa_dtype


@pytest.mark.parametrize('canvas', [
    ds.Canvas(x_axis_type='log'),
    ds.Canvas(x_axis_type='log', x_range=(0, 1)),
    ds.Canvas(y_axis_type='log'),
    ds.Canvas(y_axis_type='log', y_range=(0, 1)),
])
def test_log_axis_not_positive(ddf, canvas):
    with pytest.raises(ValueError, match='Range values must be >0 for logarithmic axes'):
        canvas.line(ddf, 'x', 'y')


@pytest.mark.parametrize('npartitions', [1, 2, 3])
def test_line_antialias_where(npartitions):
    # Identical tests to test_pandas.
    df = pd.DataFrame(dict(
        y0 = [0.5, 1.0, 0.0],
        y1 = [1.0, 0.0, 0.5],
        y2 = [0.0, 0.5, 1.0],
        value = [2.2, 3.3, 1.1],
        other = [-9.0, -7.0, -5.0],
    ))
    df = dd.from_pandas(df, npartitions=npartitions)
    assert df.npartitions == npartitions

    cvs = ds.Canvas(plot_width=7, plot_height=7)
    kwargs = dict(source=df, x=np.arange(3), y=["y0", "y1", "y2"], axis=1, line_width=1.0)

    sol_first = np.array([
        [[ 2, -1], [ 2, -1], [ 1, -1], [ 1,  1], [ 1, -1], [-1, -1], [ 0, -1]],
        [[ 2, -1], [ 2, -1], [ 1,  2], [ 1, -1], [ 1, -1], [ 0,  1], [ 0, -1]],
        [[-1, -1], [ 1,  2], [ 1,  2], [ 2, -1], [-1, -1], [ 0,  1], [ 0,  1]],
        [[ 0, -1], [ 1, -1], [ 1,  2], [ 2,  2], [ 0,  2], [ 0, -1], [ 1, -1]],
        [[ 0,  1], [ 0,  1], [-1, -1], [ 2, -1], [ 0,  2], [ 0,  2], [-1, -1]],
        [[ 1, -1], [ 0,  1], [ 0, -1], [ 0, -1], [ 0,  2], [ 2, -1], [ 2, -1]],
        [[ 1, -1], [-1, -1], [ 0, -1], [ 0,  0], [ 0, -1], [ 2, -1], [ 2, -1]]
    ], dtype=int)

    sol_last = np.array([
        [[ 2, -1], [ 2, -1], [ 1, -1], [ 1,  1], [ 1, -1], [-1, -1], [ 0, -1]],
        [[ 2, -1], [ 2, -1], [ 2,  1], [ 1, -1], [ 1, -1], [ 1,  0], [ 0, -1]],
        [[-1, -1], [ 2,  1], [ 2,  1], [ 2, -1], [-1, -1], [ 1,  0], [ 1,  0]],
        [[ 0, -1], [ 1, -1], [ 2,  1], [ 2,  2], [ 2,  0], [ 0, -1], [ 1, -1]],
        [[ 1,  0], [ 1,  0], [-1, -1], [ 2, -1], [ 2,  0], [ 2,  0], [-1, -1]],
        [[ 1, -1], [ 1,  0], [ 0, -1], [ 0, -1], [ 2,  0], [ 2, -1], [ 2, -1]],
        [[ 1, -1], [-1, -1], [ 0, -1], [ 0,  0], [ 0, -1], [ 2, -1], [ 2, -1]],
    ], dtype=int)

    sol_min = np.array([
        [[ 2, -1], [ 2, -1], [ 1, -1], [ 1,  1], [ 1, -1], [-1, -1], [ 0, -1]],
        [[ 2, -1], [ 2, -1], [ 2,  1], [ 1, -1], [ 1, -1], [ 0,  1], [ 0, -1]],
        [[-1, -1], [ 2,  1], [ 2,  1], [ 2, -1], [-1, -1], [ 0,  1], [ 0,  1]],
        [[ 0, -1], [ 1, -1], [ 2,  1], [ 2,  2], [ 2,  0], [ 0, -1], [ 1, -1]],
        [[ 1,  0], [ 0,  1], [-1, -1], [ 2, -1], [ 2,  0], [ 2,  0], [-1, -1]],
        [[ 1, -1], [ 1,  0], [ 0, -1], [ 0, -1], [ 2,  0], [ 2, -1], [ 2, -1]],
        [[ 1, -1], [-1, -1], [ 0, -1], [ 0,  0], [ 0, -1], [ 2, -1], [ 2, -1]],
    ], dtype=int)

    sol_max = np.array([
        [[ 2, -1], [ 2, -1], [ 1, -1], [ 1,  1], [ 1, -1], [-1, -1], [ 0, -1]],
        [[ 2, -1], [ 2, -1], [ 1,  2], [ 1, -1], [ 1, -1], [ 1,  0], [ 0, -1]],
        [[-1, -1], [ 1,  2], [ 1,  2], [ 2, -1], [-1, -1], [ 1,  0], [ 1,  0]],
        [[ 0, -1], [ 1, -1], [ 1,  2], [ 2,  2], [ 0,  2], [ 0, -1], [ 1, -1]],
        [[ 0,  1], [ 1,  0], [-1, -1], [ 2, -1], [ 0,  2], [ 0,  2], [-1, -1]],
        [[ 1, -1], [ 0,  1], [ 0, -1], [ 0, -1], [ 0,  2], [ 2, -1], [ 2, -1]],
        [[ 1, -1], [-1, -1], [ 0, -1], [ 0,  0], [ 0, -1], [ 2, -1], [ 2, -1]],
    ], dtype=int)

    ##### where containing first, first_n, _min_row_index and _min_n_row_index
    # where(first) returning row index then other column
    sol_index = sol_first
    sol_other = sol_index.choose(np.append(df["other"], nan), mode="wrap")

    agg = cvs.line(agg=ds.where(ds.first("value")), **kwargs)
    assert_eq_ndarray(agg.data, sol_index[:, :, 0])

    agg = cvs.line(agg=ds.where(ds.first("value"), "other"), **kwargs)
    assert_eq_ndarray(agg.data, sol_other[:, :, 0])

    # where(first_n) returning row index then other column
    agg = cvs.line(agg=ds.where(ds.first_n("value", n=2)), **kwargs)
    assert_eq_ndarray(agg.data, sol_index)

    agg = cvs.line(agg=ds.where(ds.first_n("value", n=2), "other"), **kwargs)
    assert_eq_ndarray(agg.data, sol_other)

    # where(_min_row_index) returning row index then other column
    agg = cvs.line(agg=ds.where(ds._min_row_index()), **kwargs)
    assert_eq_ndarray(agg.data, sol_index[:, :, 0])

    agg = cvs.line(agg=ds.where(ds._min_row_index(), "other"), **kwargs)
    assert_eq_ndarray(agg.data, sol_other[:, :, 0])

    # where(_min_n_row_index) returning row index then other column
    agg = cvs.line(agg=ds.where(ds._min_n_row_index(n=2)), **kwargs)
    assert_eq_ndarray(agg.data, sol_index)

    agg = cvs.line(agg=ds.where(ds._min_n_row_index(n=2), "other"), **kwargs)
    assert_eq_ndarray(agg.data, sol_other)

    ##### where containing last, last_n, _max_row_index and _max_n_row_index
    # where(last) returning row index then other column
    sol_index = sol_last
    sol_other = sol_index.choose(np.append(df["other"], nan), mode="wrap")

    agg = cvs.line(agg=ds.where(ds.last("value")), **kwargs)
    assert_eq_ndarray(agg.data, sol_index[:, :, 0])

    agg = cvs.line(agg=ds.where(ds.last("value"), "other"), **kwargs)
    assert_eq_ndarray(agg.data, sol_other[:, :, 0])

    # where(last_n) returning row index then other column
    agg = cvs.line(agg=ds.where(ds.last_n("value", n=2)), **kwargs)
    assert_eq_ndarray(agg.data, sol_index)

    agg = cvs.line(agg=ds.where(ds.last_n("value", n=2), "other"), **kwargs)
    assert_eq_ndarray(agg.data, sol_other)

    # where(_max_row_index) returning row index then other column
    agg = cvs.line(agg=ds.where(ds._max_row_index()), **kwargs)
    assert_eq_ndarray(agg.data, sol_index[:, :, 0])

    agg = cvs.line(agg=ds.where(ds._max_row_index(), "other"), **kwargs)
    assert_eq_ndarray(agg.data, sol_other[:, :, 0])

    # where(_max_n_row_index) returning row index then other column
    agg = cvs.line(agg=ds.where(ds._max_n_row_index(n=2)), **kwargs)
    assert_eq_ndarray(agg.data, sol_index)

    agg = cvs.line(agg=ds.where(ds._max_n_row_index(n=2), "other"), **kwargs)
    assert_eq_ndarray(agg.data, sol_other)

    ##### where containing min and min_n
    # where(min) returning row index then other column
    sol_index = sol_min
    sol_other = sol_index.choose(np.append(df["other"], nan), mode="wrap")

    agg = cvs.line(agg=ds.where(ds.min("value")), **kwargs)
    assert_eq_ndarray(agg.data, sol_index[:, :, 0])

    agg = cvs.line(agg=ds.where(ds.min("value"), "other"), **kwargs)
    assert_eq_ndarray(agg.data, sol_other[:, :, 0])

    # where(min_n) returning row index then other column
    agg = cvs.line(agg=ds.where(ds.min_n("value", n=2)), **kwargs)
    assert_eq_ndarray(agg.data, sol_index)

    agg = cvs.line(agg=ds.where(ds.min_n("value", n=2), "other"), **kwargs)
    assert_eq_ndarray(agg.data, sol_other)

    ##### where containing max and max_n
    # where(max) returning row index then other column
    sol_index = sol_max
    sol_other = sol_index.choose(np.append(df["other"], nan), mode="wrap")

    agg = cvs.line(agg=ds.where(ds.max("value")), **kwargs)
    assert_eq_ndarray(agg.data, sol_index[:, :, 0])

    agg = cvs.line(agg=ds.where(ds.max("value"), "other"), **kwargs)
    assert_eq_ndarray(agg.data, sol_other[:, :, 0])

    # where(max_n) returning row index then other column
    agg = cvs.line(agg=ds.where(ds.max_n("value", n=2)), **kwargs)
    assert_eq_ndarray(agg.data, sol_index)

    agg = cvs.line(agg=ds.where(ds.max_n("value", n=2), "other"), **kwargs)
    assert_eq_ndarray(agg.data, sol_other)


def test_canvas_size():
    cvs_list = [
        ds.Canvas(plot_width=0, plot_height=6),
        ds.Canvas(plot_width=5, plot_height=0),
        ds.Canvas(plot_width=0, plot_height=0),
        ds.Canvas(plot_width=-1, plot_height=1),
        ds.Canvas(plot_width=10, plot_height=-1)
    ]
    msg = r'Invalid size: plot_width and plot_height must be bigger than 0'
    df = pd.DataFrame(dict(x=[0, 0.2, 1], y=[0, 0.4, 1], z=[10, 20, 30]))
    ddf = dd.from_pandas(df, 1)

    for cvs in cvs_list:
        with pytest.raises(ValueError, match=msg):
            cvs.points(ddf, "x", "y", ds.mean("z"))


@pytest.mark.parametrize('npartitions', [1, 2, 3])
def test_dataframe_dtypes(ddf, npartitions):
    # Issue #1235.
    ddf['dates'] = pd.Series(['2007-07-13']*20, dtype='datetime64[ns]')
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    ds.Canvas(2, 2).points(ddf, 'x', 'y', ds.count())


@pytest.mark.parametrize('on_gpu', [False, pytest.param(True, marks=pytest.mark.gpu)])
def test_dask_categorical_counts(on_gpu):
    # Issue 1202
    df = pd.DataFrame(
        data=dict(
            x = [0, 1, 2, 0, 1, 2, 1, 1, 1, 1, 1, 1],
            y = [0]*12,
            cat = ['a', 'b', 'c', 'a', 'b', 'c', 'b', 'b', 'b', 'b', 'b', 'c'],
        )
    )
    ddf = dd.from_pandas(df, npartitions=2)
    assert ddf.npartitions == 2
    ddf["cat"] = ddf.cat.astype('category')

    # Categorical counts at the dataframe level to confirm test is reasonable.
    cat_totals = ddf.cat.value_counts().compute()
    assert cat_totals['a'] == 2
    assert cat_totals['b'] == 7
    assert cat_totals['c'] == 3

    canvas = ds.Canvas(3, 1, x_range=(0, 2), y_range=(-1, 1))
    agg = canvas.points(ddf, 'x', 'y', ds.by("cat", ds.count()))
    assert all(agg.cat == ['a', 'b', 'c'])

    # Prior to fix, this gives [7, 3, 2]
    sum_cat = agg.sum(dim=['x', 'y'])
    assert all(sum_cat.cat == ['a', 'b', 'c'])
    assert all(sum_cat.values == [2, 7, 3])


def test_categorical_where_max(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_rowindex = xr.DataArray([[[4, 1, -1, 3], [12, 13, 14, 11]],
                                 [[8, 5, 6, 7], [16, 17, 18, 15]]],
                                coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    # Using row index
    agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.max('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)

    # Using another column
    agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.max('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)


def test_categorical_where_min(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_rowindex = xr.DataArray([[[0, 1, -1, 3], [12, 13, 10, 11]],
                                 [[8, 9, 6, 7], [16, 17, 18, 19]]],
                                coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    # Using row index
    agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.min('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)

    # Using another column
    agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.min('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)


def test_categorical_where_first(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_rowindex = xr.DataArray([[[0, 1, -1, 3], [12, 13, 10, 11]],
                                 [[8, 5, 6, 7], [16, 17, 18, 15]]],
                                coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    # Using row index
    agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.first('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)

    # Using another column
    agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.first('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)


def test_categorical_where_last(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_rowindex = xr.DataArray([[[4, 1, -1, 3], [12, 13, 14, 11]],
                                 [[8, 9, 6, 7], [16, 17, 18, 19]]],
                                coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    # Using row index
    agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.last('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)

    # Using another column
    agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.last('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)


def test_categorical_where_max_n(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_rowindex = xr.DataArray(
        [[[[4, 0, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]],
          [[12, -1, -1], [13, -1, -1], [14, 10, -1], [11, -1, -1]]],
         [[[8, -1, -1], [5, 9, -1], [6, -1, -1], [7, -1, -1]],
          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [15, 19, -1]]]],
        coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 4):
        # Using row index
        agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.max_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(ddf, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.max('plusminus')))).data)

        # Using another column
        agg = c.points(ddf, 'x', 'y',
                       ds.by('cat2', ds.where(ds.max_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(ddf, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.max('plusminus'),
                                                              'reverse'))).data)


def test_categorical_where_min_n(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_rowindex = xr.DataArray(
        [[[[0, 4, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]],
          [[12, -1, -1], [13, -1, -1], [10, 14, -1], [11, -1, -1]]],
         [[[8, -1, -1], [9, 5, -1], [6, -1, -1], [7, -1, -1]],
          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [19, 15, -1]]]],
        coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 4):
        # Using row index
        agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.min_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(ddf, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.min('plusminus')))).data)

        # Using another column
        agg = c.points(ddf, 'x', 'y',
                       ds.by('cat2', ds.where(ds.min_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(ddf, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.min('plusminus'),
                                                              'reverse'))).data)


def test_categorical_where_first_n(ddf, npartitions):
    # Important to test with npartitions > 2 to have multiple combination stages.
    # Identical results to equivalent pandas test.
    ddf = ddf.repartition(npartitions=npartitions)
    assert ddf.npartitions == npartitions
    sol_rowindex = xr.DataArray(
        [[[[0, 4, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]],
          [[12, -1, -1], [13, -1, -1], [10, 14, -1], [11, -1, -1]]],
         [[[8, -1, -1], [5, 9, -1], [6, -1, -1], [7, -1, -1]],
          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [15, 19, -1]]]],
        coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 4):
        # Using row index
        agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.first_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(ddf, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.first('plusminus')))).data)

        # Using another column
        agg = c.points(ddf, 'x', 'y',
                       ds.by('cat2', ds.where(ds.first_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(ddf, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.first('plusminus'),
                                                              'reverse'))).data)


def test_categorical_where_last_n(ddf, npartitions):
    sol_rowindex = xr.DataArray(
        [[[[4, 0, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]],
          [[12, -1, -1], [13, -1, -1], [14, 10, -1], [11, -1, -1]]],
         [[[8, -1, -1], [9, 5, -1], [6, -1, -1], [7, -1, -1]],
          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [19, 15, -1]]]],
        coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan,
                           20 - sol_rowindex)

    for n in range(1, 4):
        # Using row index
        agg = c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.last_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(ddf, 'x', 'y',
                                       ds.by('cat2', ds.where(ds.last('plusminus')))).data)

        # Using another column
        agg = c.points(ddf, 'x', 'y',
                       ds.by('cat2', ds.where(ds.last_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(ddf, 'x', 'y', ds.by('cat2', ds.where(ds.last('plusminus'),
                                                                             'reverse'))).data)

def test_series_reset_index(ddf, npartitions):
    # Test for: https://github.com/holoviz/datashader/issues/1331
    ser = ddf['i32'].reset_index()
    cvs = ds.Canvas(plot_width=2, plot_height=2)
    out = cvs.line(ser, x='index', y='i32')

    expected = xr.DataArray(
        data=[[True, False], [False, True]],
        coords={"index": [4.75, 14.25], "i32": [4.75, 14.25]},
        dims=['i32', 'index'],
    )
    assert_eq_xr(out, expected)
