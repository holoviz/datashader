from __future__ import annotations
import os
from numpy import nan

import numpy as np
import pandas as pd
import xarray as xr

import datashader as ds

import pytest

from datashader.datatypes import RaggedDtype

df_pd = pd.DataFrame({'x': np.array(([0.] * 10 + [1] * 10)),
                      'y': np.array(([0.] * 5 + [1] * 5 + [0] * 5 + [1] * 5)),
                      'log_x': np.array(([1.] * 10 + [10] * 10)),
                      'log_y': np.array(([1.] * 5 + [10] * 5 + [1] * 5 + [10] * 5)),
                      'i32': np.arange(20, dtype='i4'),
                      'i64': np.arange(20, dtype='i8'),
                      'f32': np.arange(20, dtype='f4'),
                      'f64': np.arange(20, dtype='f8'),
                      'reverse': np.arange(20, 0, -1),
                      'plusminus': np.arange(20, dtype='f8')*([1, -1]*10),
                      'empty_bin': np.array([0.] * 15 + [np.nan] * 5),
                      'cat': ['a']*5 + ['b']*5 + ['c']*5 + ['d']*5,
                      'cat2': ['a', 'b', 'c', 'd']*5,
                      'onecat': ['one']*20,
                      'cat_int': np.array([10]*5 + [11]*5 + [12]*5 + [13]*5)})
df_pd.cat = df_pd.cat.astype('category')
df_pd.cat2 = df_pd.cat2.astype('category')
df_pd.onecat = df_pd.onecat.astype('category')
df_pd.at[2, 'f32'] = nan
df_pd.at[2, 'f64'] = nan
df_pd.at[6, 'reverse'] = nan
df_pd.at[2, 'plusminus'] = nan
# x          0  0   0  0  0   0   0  0  0  0   1   1  1   1  1    1  1   1  1   1
# y          0  0   0  0  0   1   1  1  1  1   0   0  0   0  0    1  1   1  1   1
# i32        0  1   2  3  4   5   6  7  8  9  10  11 12  13 14   15 16  17 18  19
# f32        0  1 nan  3  4   5   6  7  8  9  10  11 12  13 14   15 16  17 18  19
# reverse   20 19  18 17 16  15 nan 13 12 11  10   9  8   7  6    5  4   3  2   1
# plusminus  0 -1 nan -3  4  -5   6 -7  8 -9  10 -11 12 -13 14  -15 16 -17 18 -19
# cat2       a  b   c  d  a   b   c  d  a  b   c   d  a   b  c    d  a   b  c   d

test_gpu = bool(int(os.getenv("DATASHADER_TEST_GPU", 0)))


try:
    import spatialpandas as sp
    from spatialpandas.geometry import LineDtype
except ImportError:
    LineDtype = None
    sp = None


def pd_DataFrame(*args, **kwargs):
    if kwargs.pop("geo", False):
        return sp.GeoDataFrame(*args, **kwargs)
    else:
        return pd.DataFrame(*args, **kwargs)


try:
    import cudf
    import cupy

    if not test_gpu:
        # GPU testing disabled even though cudf/cupy are available
        raise ImportError

    def cudf_DataFrame(*args, **kwargs):
        assert not kwargs.pop("geo", False)
        return cudf.DataFrame.from_pandas(
            pd.DataFrame(*args, **kwargs), nan_as_null=False
        )
    df_cuda = cudf_DataFrame(df_pd)
    dfs = [df_pd, df_cuda]
    DataFrames = [pd_DataFrame, cudf_DataFrame]
except ImportError:
    cudf = cupy = None
    dfs = [df_pd]
    DataFrames = [pd_DataFrame]


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


def assert_eq_xr(agg, b, close=False):
    """Assert that two xarray DataArrays are equal, handling the possibility
    that the two DataArrays may be backed by ndarrays of different types"""
    if cupy:
        if isinstance(agg.data, cupy.ndarray):
            agg = xr.DataArray(
                cupy.asnumpy(agg.data), coords=agg.coords, dims=agg.dims
            )
        if isinstance(b.data, cupy.ndarray):
            b = xr.DataArray(
                cupy.asnumpy(b.data), coords=b.coords, dims=b.dims
            )
    if close:
        xr.testing.assert_allclose(agg, b)
    else:
        xr.testing.assert_equal(agg, b)

def assert_eq_ndarray(data, b, close=False):
    """Assert that two ndarrays are equal, handling the possibility that the
    ndarrays are of different types"""
    if cupy:
        if isinstance(data, cupy.ndarray):
            data = cupy.asnumpy(data)
        if isinstance(b, cupy.ndarray):
            b = cupy.asnumpy(b)

    if close:
        np.testing.assert_array_almost_equal(data, b, decimal=5)
    else:
        np.testing.assert_equal(data, b)


def assert_image_close(image0, image1, tolerance):
    def to_rgba_xr(image):
        data = image.data
        if cupy and isinstance(data, cupy.ndarray):
            data = cupy.asnumpy(data)
        shape = data.shape
        data = data.view(np.uint8).reshape(shape + (4,))
        return xr.DataArray(data, dims=image.dims + ("rgba",), coords=image.coords)

    da0 = to_rgba_xr(image0)
    da1 = to_rgba_xr(image1)
    xr.testing.assert_allclose(da0, da1, atol=tolerance, rtol=0)


def floats(n):
    """Returns contiguous list of floats from initial point"""
    while True:
        yield n
        n = n + np.spacing(n)


def values(s):
    """Get numpy array of values from pandas-like Series, handling Series
    of different types"""
    if cudf and isinstance(s, cudf.Series):
        try:
            return s.to_numpy(na_value=np.nan)
        except AttributeError:
            # to_array is deprecated from cudf 22.02
            return s.to_array(fillna=np.nan)
    else:
        return s.values


def test_gpu_dependencies():
    if test_gpu and cudf is None:
        pytest.fail("cudf and/or cupy not available and DATASHADER_TEST_GPU=1")


@pytest.mark.skipif(not test_gpu, reason="DATASHADER_TEST_GPU not set")
def test_cudf_concat():
    # Testing if a newer version of cuDF implements the possibility to
    # concatenate multiple columns with the same name.
    # Currently, a workaround for this is
    # implemented in `datashader.glyphs.Glyph.to_cupy_array`.
    # For details, see: https://github.com/holoviz/datashader/pull/1050

    with pytest.raises(NotImplementedError):
        dfp = pd.DataFrame({'y': [0, 1]})
        dfc = cudf.from_pandas(dfp)
        cudf.concat((dfc["y"], dfc["y"]), axis=1)


@pytest.mark.parametrize('df', dfs)
def test_count(df):
    out = xr.DataArray(np.array([[5, 5], [5, 5]], dtype='i4'),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.count('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.count('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.count()), out)
    out = xr.DataArray(np.array([[4, 5], [5, 5]], dtype='i4'),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.count('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.count('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_any(df):
    out = xr.DataArray(np.array([[True, True], [True, True]]),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.any('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.any('f64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.any()), out)
    out = xr.DataArray(np.array([[True, True], [True, False]]),
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.any('empty_bin')), out)


@pytest.mark.parametrize('df', dfs)
def test_sum(df):
    out = xr.DataArray(values(df.i32).reshape((2, 2, 5)).sum(axis=2, dtype='f8').T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.sum('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.sum('i64')), out)
    out = xr.DataArray(np.nansum(values(df.f64).reshape((2, 2, 5)), axis=2).T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.sum('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.sum('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_min(df):
    out = xr.DataArray(values(df.i64).reshape((2, 2, 5)).min(axis=2).astype('f8').T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.min('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.min('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.min('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.min('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_max(df):
    out = xr.DataArray(values(df.i64).reshape((2, 2, 5)).max(axis=2).astype('f8').T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.max('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.max('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.max('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.max('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_min_n(df):
    solution = np.array([[[-3, -1, 0, 4, nan, nan], [-13, -11, 10, 12, 14, nan]],
                         [[-9, -7, -5, 6, 8, nan], [-19, -17, -15, 16, 18, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.min_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.min('plusminus')).data)


@pytest.mark.parametrize('df', dfs)
def test_max_n(df):
    solution = np.array([[[4, 0, -1, -3, nan, nan], [14, 12, 10, -11, -13, nan]],
                         [[8, 6, -5, -7, -9, nan], [18, 16, -15, -17, -19, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.max_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.max('plusminus')).data)


@pytest.mark.parametrize('df', dfs)
def test_categorical_min(df):
    sol_int = np.array([[[0, 1, 2, 3], [12, 13, 10, 11]], [[8, 5, 6, 7], [16, 17, 18, 15]]], dtype=np.float64)
    sol_float = np.array([[[0, 1, nan, 3], [12, 13, 10, 11]], [[8, 5, 6, 7], [16, 17, 18, 15]]])
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.min('i32'))).data, sol_int)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.min('i64'))).data, sol_int)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.min('f32'))).data, sol_float)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.min('f64'))).data, sol_float)


@pytest.mark.parametrize('df', dfs)
def test_categorical_max(df):
    sol_int = np.array([[[4, 1, 2, 3], [12, 13, 14, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]], dtype=np.float64)
    sol_float = np.array([[[4, 1, nan, 3], [12, 13, 14, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]])
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.max('i32'))).data, sol_int)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.max('i64'))).data, sol_int)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.max('f32'))).data, sol_float)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.max('f64'))).data, sol_float)


@pytest.mark.parametrize('df', dfs)
def test_categorical_min_n(df):
    solution = np.array([[[[0, 4, nan], [1, nan, nan], [nan, nan, nan], [3, nan, nan]],
                          [[12, nan, nan], [13, nan, nan], [10, 14, nan], [11, nan, nan]]],
                         [[[8, nan, nan], [5, 9, nan], [6, nan, nan], [7, nan, nan]],
                          [[16, nan, nan], [17, nan, nan], [18, nan, nan], [15, 19, nan]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.min_n('f32', n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data, c.points(df, 'x', 'y', ds.by('cat2', ds.min('f32'))).data)


@pytest.mark.parametrize('df', dfs)
def test_categorical_max_n(df):
    solution = np.array([[[[4, 0, nan], [1, nan, nan], [nan, nan, nan], [3, nan, nan]],
                          [[12, nan, nan], [13, nan, nan], [14, 10, nan], [11, nan, nan]]],
                         [[[8, nan, nan], [9, 5, nan], [6, nan, nan], [7, nan, nan]],
                          [[16, nan, nan], [17, nan, nan], [18, nan, nan], [19, 15, nan]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.max_n('f32', n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data, c.points(df, 'x', 'y', ds.by('cat2', ds.max('f32'))).data)


@pytest.mark.parametrize('df', dfs)
def test_categorical_min_row_index(df):
    solution = np.array([[[0, 1, 2, 3], [12, 13, 10, 11]], [[8, 5, 6, 7], [16, 17, 18, 15]]])
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds._min_row_index()))
    assert_eq_ndarray(agg.data, solution)


@pytest.mark.parametrize('df', dfs)
def test_categorical_max_row_index(df):
    solution = np.array([[[4, 1, 2, 3], [12, 13, 14, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]])
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds._max_row_index()))
    assert_eq_ndarray(agg.data, solution)


@pytest.mark.parametrize('df', dfs)
def test_categorical_min_n_row_index(df):
    solution = np.array([[[[0, 4, -1], [1, -1, -1], [2, -1, -1], [3, -1, -1]],
                          [[12, -1, -1], [13, -1, -1], [10, 14, -1], [11, -1, -1]]],
                         [[[8, -1, -1], [5, 9, -1], [6, -1, -1], [7, -1, -1]],
                          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [15, 19, -1]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds._min_n_row_index(n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data, c.points(df, 'x', 'y', ds.by('cat2', ds._min_row_index())).data)


@pytest.mark.parametrize('df', dfs)
def test_categorical_max_n_row_index(df):
    solution = np.array([[[[4, 0, -1], [1, -1, -1], [2, -1, -1], [3, -1, -1]],
                          [[12, -1, -1], [13, -1, -1], [14, 10, -1], [11, -1, -1]]],
                         [[[8, -1, -1], [9, 5, -1], [6, -1, -1], [7, -1, -1]],
                          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [19, 15, -1]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds._max_n_row_index(n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data, c.points(df, 'x', 'y', ds.by('cat2', ds._max_row_index())).data)


@pytest.mark.parametrize('df', dfs)
def test_categorical_first(df):
    solution = np.array([[[0, -1, nan, -3],
                          [12, -13, 10, -11]],
                         [[8, -5, 6, -7],
                          [16, -17, 18, -15]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.first("plusminus")))
        assert_eq_ndarray(agg.data, solution)


@pytest.mark.parametrize('df', dfs)
def test_categorical_last(df):
    solution = np.array([[[4, -1, nan, -3],
                          [12, -13, 14, -11]],
                         [[8, -9, 6, -7],
                          [16, -17, 18, -19]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.last("plusminus")))
        assert_eq_ndarray(agg.data, solution)


@pytest.mark.parametrize('df', dfs)
def test_categorical_first_n(df):
    solution = np.array([[[[0, 4, nan], [-1, nan, nan], [nan, nan, nan], [-3, nan, nan]],
                          [[12, nan, nan], [-13, nan, nan], [10, 14, nan], [-11, nan, nan]]],
                         [[[8, nan, nan], [-5, -9, nan], [6, nan, nan], [-7, nan, nan]],
                          [[16, nan, nan], [-17, nan, nan], [18, nan, nan], [-15, -19, nan]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.first_n("plusminus", n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data, c.points(df, 'x', 'y', ds.by('cat2', ds.first("plusminus"))).data)


@pytest.mark.parametrize('df', dfs)
def test_categorical_last_n(df):
    solution = np.array([[[[4, 0, nan], [-1, nan, nan], [nan, nan, nan], [-3, nan, nan]],
                          [[12, nan, nan], [-13, nan, nan], [14, 10, nan], [-11, nan, nan]]],
                         [[[8, nan, nan], [-9, -5, nan], [6, nan, nan], [-7, nan, nan]],
                          [[16, nan, nan], [-17, nan, nan], [18, nan, nan], [-19, -15, nan]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.last_n("plusminus", n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data, c.points(df, 'x', 'y', ds.by('cat2', ds.last("plusminus"))).data)


@pytest.mark.parametrize('df', dfs)
def test_where_min_row_index(df):
    out = xr.DataArray([[0, 10], [-5, -15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds._min_row_index(), 'plusminus')), out)


@pytest.mark.parametrize('df', dfs)
def test_where_max_row_index(df):
    out = xr.DataArray([[4, 14], [-9, -19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds._max_row_index(), 'plusminus')), out)


@pytest.mark.parametrize('df', dfs)
def test_where_min_n_row_index(df):
    sol = np.array([[[  0,  -1, nan,  -3,   4, nan],
                     [ 10, -11,  12, -13,  14, nan]],
                    [[ -5,   6,  -7,   8,  -9, nan],
                     [-15,  16, -17,  18, -19, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.where(ds._min_n_row_index(n=n), 'plusminus'))
        out = sol[:, :, :n]
        print(n, agg.data.tolist())
        print(' ', out.tolist())
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds._min_row_index(), 'plusminus')).data)


@pytest.mark.parametrize('df', dfs)
def test_where_max_n_row_index(df):
    sol = np.array([[[  4,  -3, nan,  -1,   0, nan],
                     [ 14, -13,  12, -11,  10, nan]],
                    [[ -9,   8,  -7,   6,  -5, nan],
                     [-19,  18, -17,  16, -15, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.where(ds._max_n_row_index(n=n), 'plusminus'))
        out = sol[:, :, :n]
        print(n, agg.data.tolist())
        print(' ', out.tolist())
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds._max_row_index(), 'plusminus')).data)


@pytest.mark.parametrize('df', dfs)
def test_where_first(df):
    # Note reductions like ds.where(ds.first('i32'), 'reverse') are supported,
    # but the same results can be achieved using the simpler ds.first('reverse')
    out = xr.DataArray([[20, 10], [15, 5]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('i32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('i64'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('f32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('f64'), 'reverse')), out)

    # Using row index.
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('i32'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('i64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('f64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.first('f32'))), out)


@pytest.mark.parametrize('df', dfs)
def test_where_last(df):
    # Note reductions like ds.where(ds.last('i32'), 'reverse') are supported,
    # but the same results can be achieved using the simpler ds.last('reverse')
    out = xr.DataArray([[16, 6], [11, 1]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('i32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('i64'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('f32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('f64'), 'reverse')), out)

    # Using row index.
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('i32'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('i64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('f64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('f32'))), out)


@pytest.mark.parametrize('df', dfs)
def test_where_max(df):
    out = xr.DataArray([[16, 6], [11, 1]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('i32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('i64'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('f32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('f64'), 'reverse')), out)

    # Using row index.
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('i32'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('i64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('f64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.max('f32'))), out)


@pytest.mark.parametrize('df', dfs)
def test_where_min(df):
    out = xr.DataArray([[20, 10], [15, 5]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('i32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('i64'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('f32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('f64'), 'reverse')), out)

    # Using row index.
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('i32'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('i64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('f64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('f32'))), out)


@pytest.mark.parametrize('df', dfs)
def test_where_first_n(df):
    sol_rowindex = np.array([[[ 0,  1,  3,  4, -1, -1],
                              [10, 11, 12, 13, 14, -1]],
                             [[ 5,  6,  7,  8,  9, -1],
                              [15, 16, 17, 18, 19, -1]]])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)

    for n in range(1, 7):
        # Using row index.
        agg = c.points(df, 'x', 'y', ds.where(ds.first_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.first('plusminus'))).data)

        # Using another column
        agg = c.points(df, 'x', 'y', ds.where(ds.first_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.first('plusminus'), 'reverse')).data)


@pytest.mark.parametrize('df', dfs)
def test_where_last_n(df):
    sol_rowindex = np.array([[[ 4,  3,  1,  0, -1, -1],
                              [14, 13, 12, 11, 10, -1]],
                             [[ 9,  8,  7,  6,  5, -1],
                              [19, 18, 17, 16, 15, -1]]])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)

    for n in range(1, 7):
        # Using row index.
        agg = c.points(df, 'x', 'y', ds.where(ds.last_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.last('plusminus'))).data)

        # Using another column
        agg = c.points(df, 'x', 'y', ds.where(ds.last_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.last('plusminus'), 'reverse')).data)


@pytest.mark.parametrize('df', dfs)
def test_where_max_n(df):
    sol_rowindex = np.array([[[ 4,  0,  1,  3, -1, -1],
                              [14, 12, 10, 11, 13, -1]],
                             [[ 8,  6,  5,  7,  9, -1],
                              [18, 16, 15, 17, 19, -1]]])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)

    for n in range(1, 7):
        # Using row index.
        agg = c.points(df, 'x', 'y', ds.where(ds.max_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.max('plusminus'))).data)

        # Using another column
        agg = c.points(df, 'x', 'y', ds.where(ds.max_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.max('plusminus'), 'reverse')).data)


@pytest.mark.parametrize('df', dfs)
def test_where_min_n(df):
    sol_rowindex = np.array([[[3,  1,  0,  4, -1, -1],
                              [13, 11, 10, 12, 14, -1]],
                             [[ 9,  7,  5,  6,  8, -1],
                              [19, 17, 15, 16, 18, -1]]])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)

    for n in range(1, 7):
        # Using row index.
        agg = c.points(df, 'x', 'y', ds.where(ds.min_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.min('plusminus'))).data)

        # Using another column
        agg = c.points(df, 'x', 'y', ds.where(ds.min_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.min('plusminus'), 'reverse')).data)


@pytest.mark.parametrize('df', dfs)
def test_summary_by(df):
    # summary(by)
    agg_summary = c.points(df, 'x', 'y', ds.summary(by=ds.by("cat")))
    agg_by = c.points(df, 'x', 'y', ds.by("cat"))
    assert_eq_xr(agg_summary["by"], agg_by)

    # summary(by, other_reduction)
    agg_summary = c.points(df, 'x', 'y', ds.summary(by=ds.by("cat"), max=ds.max("plusminus")))
    agg_max = c.points(df, 'x', 'y', ds.max("plusminus"))
    assert_eq_xr(agg_summary["by"], agg_by)
    assert_eq_xr(agg_summary["max"], agg_max)

    # summary(other_reduction, by)
    agg_summary = c.points(df, 'x', 'y', ds.summary(max=ds.max("plusminus"), by=ds.by("cat")))
    assert_eq_xr(agg_summary["by"], agg_by)
    assert_eq_xr(agg_summary["max"], agg_max)

    # summary(by, by)
    agg_summary = c.points(df, 'x', 'y', ds.summary(by=ds.by("cat"), by_any=ds.by("cat", ds.any())))
    agg_by_any = c.points(df, 'x', 'y', ds.by("cat", ds.any()))
    assert_eq_xr(agg_summary["by"], agg_by)
    assert_eq_xr(agg_summary["by_any"], agg_by_any)

    # summary(by("cat1"), by("cat2"))
    agg_summary = c.points(df, 'x', 'y', ds.summary(by=ds.by("cat"), by2=ds.by("cat2")))
    agg_by2 = c.points(df, 'x', 'y', ds.by("cat2"))
    assert_eq_xr(agg_summary["by"], agg_by)
    assert_eq_xr(agg_summary["by2"], agg_by2)


@pytest.mark.parametrize('df', dfs)
def test_summary_where_n(df):
    sol_min_n_rowindex = np.array([[[ 3,  1,  0,  4, -1],
                                    [13, 11, 10, 12, 14]],
                                   [[ 9,  7,  5,  6,  8],
                                    [19, 17, 15, 16, 18]]])
    sol_max_n_rowindex = np.array([[[ 4,  0,  1,  3, -1],
                                    [14, 12, 10, 11, 13]],
                                   [[ 8,  6,  5,  7,  9],
                                    [18, 16, 15, 17, 19]]])
    sol_max_n_reverse = np.where(np.logical_or(sol_max_n_rowindex < 0, sol_max_n_rowindex == 6),
                                 np.nan, 20 - sol_max_n_rowindex)

    agg = c.points(df, 'x', 'y', ds.summary(
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

    # Issue #1270: Support summary reduction containing multiple where
    # reductions that use the same selector.
    agg = c.points(df, 'x', 'y', ds.summary(
        max1=ds.where(ds.max_n('plusminus', 5)),
        max2=ds.where(ds.max_n('plusminus', 5), 'reverse'),
    ))
    assert_eq_ndarray(agg['max1'].data, sol_max_n_rowindex)
    assert_eq_ndarray(agg['max2'].data, sol_max_n_reverse)


@pytest.mark.parametrize('df', dfs)
def test_summary_different_n(df):
    msg = 'Using multiple FloatingNReductions with different n values is not supported'
    with pytest.raises(ValueError, match=msg):
        c.points(df, 'x', 'y', ds.summary(
            min_n=ds.where(ds.min_n('plusminus', 2)),
            max_n=ds.where(ds.max_n('plusminus', 3)),
        ))


@pytest.mark.parametrize('df', dfs)
def test_mean(df):
    out = xr.DataArray(values(df.i32).reshape((2, 2, 5)).mean(axis=2, dtype='f8').T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.mean('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.mean('i64')), out)
    out = xr.DataArray(np.nanmean(values(df.f64).reshape((2, 2, 5)), axis=2).T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.mean('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.mean('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_var(df):
    out = xr.DataArray(values(df.i32).reshape((2, 2, 5)).var(axis=2, dtype='f8').T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.var('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.var('i64')), out)
    out = xr.DataArray(np.nanvar(values(df.f64).reshape((2, 2, 5)), axis=2).T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.var('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.var('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_std(df):
    out = xr.DataArray(values(df.i32).reshape((2, 2, 5)).std(axis=2, dtype='f8').T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.std('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.std('i64')), out)
    out = xr.DataArray(np.nanstd(values(df.f64).reshape((2, 2, 5)), axis=2).T,
                       coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.std('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.std('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_count_cat(df):
    sol = np.array([[[5, 0, 0, 0],
                     [0, 0, 5, 0]],
                    [[0, 5, 0, 0],
                     [0, 0, 0, 5]]])
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=(dims + ['cat']))
    agg = c.points(df, 'x', 'y', ds.count_cat('cat'))
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 1), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)


@pytest.mark.parametrize('df', dfs)
def test_categorical_count(df):
    sol = np.array([[[5, 0, 0, 0],
                     [0, 0, 5, 0]],
                    [[0, 5, 0, 0],
                     [0, 0, 0, 5]]])
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=(dims + ['cat']))
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.count('i32')))
    assert_eq_xr(agg, out)

    # ds.summary(name=ds.by("cat")) should give same result as ds.by("cat"). Issue 1252
    dataset = c.points(df, 'x', 'y', ds.summary(name=ds.by('cat', ds.count('i32'))))
    assert_eq_xr(dataset["name"], out)

    # categorizing by (cat_int-10)%4 ought to give the same result
    out = xr.DataArray(sol, coords=coords + [range(4)], dims=(dims + ['cat_int']))
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.count()))
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_one_category(df):
    # Issue #1142.
    assert len(df['onecat'].unique()) == 1
    sol = np.array([[[5], [5]], [[5], [5]]])
    out = xr.DataArray(sol, coords=coords + [['one']], dims=(dims + ['onecat']))
    agg = c.points(df, 'x', 'y', ds.by('onecat', ds.count('i32')))
    assert agg.shape == (2, 2, 1)
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_categorical_count_binning(df):
    sol = np.array([[[5, 0, 0, 0],
                     [0, 0, 5, 0]],
                    [[0, 5, 0, 0],
                     [0, 0, 0, 5]]])

    # add an extra category (this will count nans and out of bounds)
    sol = np.append(sol, [[[0], [0]],[[0], [0]]], axis=2)

    # categorizing by binning the integer arange columns using [0,20] into 4 bins. Same result as for count_cat
    for col in 'i32', 'i64':
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=(dims + [col]))
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.count()))
        assert_eq_xr(agg, out)

    # as above, but for the float arange columns. Element 2 has a nan, so the first bin is one short, and the nan bin is +1
    sol[0, 0, 0] = 4
    sol[0, 0, 4] = 1

    for col in 'f32', 'f64':
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=(dims + [col]))
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.count()))
        assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_categorical_sum(df):
    sol = np.array([[[ 10, nan, nan, nan],
                     [nan, nan,  60, nan]],
                    [[nan,  35, nan, nan],
                     [nan, nan, nan,  85]]])
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=(dims + ['cat']))
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.sum('i32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.sum('i64')))
    assert_eq_xr(agg, out)

    # categorizing by (cat_int-10)%4 ought to give the same result
    out = xr.DataArray(sol, coords=coords + [range(4)], dims=(dims + ['cat_int']))

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.sum('i32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.sum('i64')))
    assert_eq_xr(agg, out)

    sol = np.array([[[8.0,  nan,  nan,  nan],
                     [nan,  nan, 60.0,  nan]],
                    [[nan, 35.0,  nan,  nan],
                     [nan,  nan,  nan, 85.0]]])
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=(dims + ['cat']))
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.sum('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.sum('f64')))
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_categorical_sum_binning(df):
    sol = np.array([[[8.0,  nan,  nan,  nan],
                     [nan,  nan, 60.0,  nan]],
                    [[nan, 35.0,  nan,  nan],
                     [nan,  nan,  nan, 85.0]]])

    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=(dims + [col]))
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.sum(col)))
        assert_eq_xr(agg, out)
        assert_eq_ndarray(agg.x_range, (0, 1), close=True)
        assert_eq_ndarray(agg.y_range, (0, 1), close=True)


@pytest.mark.parametrize('df', dfs)
def test_categorical_max2(df):
    sol = np.array([[[  4, nan, nan, nan],
                     [nan, nan,  14, nan]],
                    [[nan,   9, nan, nan],
                     [nan, nan, nan,  19]]])
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=(dims + ['cat']))
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.max('i32')))
    assert_eq_xr(agg, out)

    # categorizing by (cat_int-10)%4 ought to give the same result
    out = xr.DataArray(sol, coords=coords + [range(4)], dims=(dims + ['cat_int']))

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.max('i32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.max('i64')))
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_categorical_max_binning(df):
    sol = np.array([[[  4, nan, nan, nan],
                     [nan, nan,  14, nan]],
                    [[nan,   9, nan, nan],
                     [nan, nan, nan,  19]]])

    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=(dims + [col]))
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.max(col)))
        assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_categorical_mean(df):
    sol = np.array([[[  2, nan, nan, nan],
                     [nan, nan,  12, nan]],
                    [[nan,   7, nan, nan],
                     [nan, nan, nan,  17]]])
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=(dims + ['cat']))

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.mean('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.mean('f64')))
    assert_eq_xr(agg, out)

    # categorizing by (cat_int-10)%4 ought to give the same result
    out = xr.DataArray(sol, coords=coords + [range(4)], dims=(dims + ['cat_int']))

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.mean('i32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.mean('i64')))
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_categorical_mean_binning(df):
    sol = np.array([[[  2, nan, nan, nan],
                     [nan, nan,  12, nan]],
                    [[nan,   7, nan, nan],
                     [nan, nan, nan,  17]]])

    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=(dims + [col]))
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.mean(col)))
        assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_categorical_var(df):
    sol = np.array([[[ 2.5,  nan,  nan,  nan],
                     [ nan,  nan,   2.,  nan]],
                    [[ nan,   2.,  nan,  nan],
                     [ nan,  nan,  nan,   2.]]])
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=(dims + ['cat']))

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.var('f32')))
    assert_eq_xr(agg, out, True)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.var('f64')))
    assert_eq_xr(agg, out, True)

        # categorizing by (cat_int-10)%4 ought to give the same result
    out = xr.DataArray(sol, coords=coords + [range(4)], dims=(dims + ['cat_int']))

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.var('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.var('f64')))
    assert_eq_xr(agg, out)

    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=(dims + [col]))
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.var(col)))
        assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_categorical_std(df):
    sol = np.sqrt(np.array([
        [[ 2.5,  nan,  nan,  nan],
         [ nan,  nan,   2.,  nan]],
        [[ nan,   2.,  nan,  nan],
         [ nan,  nan,  nan,   2.]]])
    )
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=(dims + ['cat']))

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.std('f32')))
    assert_eq_xr(agg, out, True)

    agg = c.points(df, 'x', 'y', ds.by('cat', ds.std('f64')))
    assert_eq_xr(agg, out, True)

    # categorizing by (cat_int-10)%4 ought to give the same result
    out = xr.DataArray(sol, coords=coords + [range(4)], dims=(dims + ['cat_int']))

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.std('f32')))
    assert_eq_xr(agg, out)

    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.std('f64')))
    assert_eq_xr(agg, out)

    sol = np.append(sol, [[[nan], [nan]],[[nan], [nan]]], axis=2)

    for col in 'f32', 'f64':
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=(dims + [col]))
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.std(col)))
        assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_first(df):
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.first('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.first('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.first('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.first('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_last(df):
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.last('i32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.last('i64')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.last('f32')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.last('f64')), out)


@pytest.mark.parametrize('df', dfs)
def test_first_n(df):
    solution = np.array([[[0, -1, -3, 4, nan, nan], [10, -11, 12, -13, 14, nan]],
                         [[-5, 6, -7, 8, -9, nan], [-15, 16, -17, 18, -19, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.first_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.first('plusminus')).data)


@pytest.mark.parametrize('df', dfs)
def test_last_n(df):
    solution = np.array([[[4, -3, -1, 0, nan, nan], [14, -13, 12, -11, 10, nan]],
                         [[-9, 8, -7, 6, -5, nan], [-19, 18, -17, 16, -15, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.last_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.last('plusminus')).data)


@pytest.mark.parametrize('df', dfs)
def test_min_row_index(df):
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    agg = c.points(df, 'x', 'y', ds._min_row_index())
    assert agg.dtype == np.int64
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_max_row_index(df):
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    agg = c.points(df, 'x', 'y', ds._max_row_index())
    assert agg.dtype == np.int64
    assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_min_n_row_index(df):
    solution = np.array([[[0, 1, 2, 3, 4, -1], [10, 11, 12, 13, 14, -1]],
                         [[5, 6, 7, 8, 9, -1], [15, 16, 17, 18, 19, -1]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds._min_n_row_index(n=n))
        assert agg.dtype == np.int64
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds._min_row_index()).data)


@pytest.mark.parametrize('df', dfs)
def test_max_n_row_index(df):
    solution = np.array([[[4, 3, 2, 1, 0, -1], [14, 13, 12, 11, 10, -1]],
                         [[9, 8, 7, 6, 5, -1], [19, 18, 17, 16, 15, -1]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds._max_n_row_index(n=n))
        assert agg.dtype == np.int64
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds._max_row_index()).data)


@pytest.mark.parametrize('df', dfs)
def test_multiple_aggregates(df):
    agg = c.points(df, 'x', 'y',
                   ds.summary(f64_mean=ds.mean('f64'),
                              i32_sum=ds.sum('i32'),
                              i32_count=ds.count('i32')))

    f = lambda x: xr.DataArray(x, coords=coords, dims=dims)
    assert_eq_xr(agg.f64_mean, f(np.nanmean(values(df.f64).reshape((2, 2, 5)), axis=2).T))
    assert_eq_xr(agg.i32_sum, f(values(df.i32).reshape((2, 2, 5)).sum(axis=2, dtype='f8').T))
    assert_eq_xr(agg.i32_count, f(np.array([[5, 5], [5, 5]], dtype='i4')))


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_auto_range_points(DataFrame):
    n = 10
    data = np.arange(n, dtype='i4')
    df = DataFrame({'time': np.arange(n),
                    'x': data,
                    'y': data})

    cvs = ds.Canvas(plot_width=n, plot_height=n)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((n, n), int)
    np.fill_diagonal(sol, 1)
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 9), close=True)
    assert_eq_ndarray(agg.y_range, (0, 9), close=True)

    cvs = ds.Canvas(plot_width=n+1, plot_height=n+1)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((n+1, n+1), int)
    np.fill_diagonal(sol, 1)
    sol[5, 5] = 0
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 9), close=True)
    assert_eq_ndarray(agg.y_range, (0, 9), close=True)

    n = 4
    data = np.arange(n, dtype='i4')
    df = DataFrame({'time': np.arange(n),
                    'x': data,
                    'y': data})

    cvs = ds.Canvas(plot_width=2*n, plot_height=2*n)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((2*n, 2*n), int)
    np.fill_diagonal(sol, 1)
    sol[np.array([tuple(range(1, 4, 2))])] = 0
    sol[np.array([tuple(range(4, 8, 2))])] = 0
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 3), close=True)
    assert_eq_ndarray(agg.y_range, (0, 3), close=True)

    cvs = ds.Canvas(plot_width=2*n+1, plot_height=2*n+1)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((2*n+1, 2*n+1), int)
    sol[0, 0] = 1
    sol[3, 3] = 1
    sol[6, 6] = 1
    sol[8, 8] = 1
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 3), close=True)
    assert_eq_ndarray(agg.y_range, (0, 3), close=True)


def test_uniform_points():
    n = 101
    df = pd.DataFrame({'time': np.ones(2*n, dtype='i4'),
                       'x': np.concatenate((np.arange(n, dtype='f8'),
                                            np.arange(n, dtype='f8'))),
                       'y': np.concatenate(([0.] * n, [1.] * n))})

    cvs = ds.Canvas(plot_width=10, plot_height=2, y_range=(0, 1))
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.array([[10] * 9 + [11], [10] * 9 + [11]], dtype='i4')
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 100), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)


@pytest.mark.parametrize('high', [9, 10, 99, 100])
@pytest.mark.parametrize('low', [0])
def test_uniform_diagonal_points(low, high):
    bounds = (low, high)
    x_range, y_range = bounds, bounds

    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    n = width * height
    df = pd.DataFrame({'time': np.ones(n, dtype='i4'),
                       'x': np.array([np.arange(*x_range, dtype='f8')] * width).flatten(),
                       'y': np.array([np.arange(*y_range, dtype='f8')] * height).flatten()})

    cvs = ds.Canvas(plot_width=2, plot_height=2, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))

    diagonal = agg.data.diagonal(0)
    assert sum(diagonal) == n
    assert abs(bounds[1] - bounds[0]) % 2 == abs(diagonal[1] / high - diagonal[0] / high)

    assert_eq_ndarray(agg.x_range, (low, high), close=True)
    assert_eq_ndarray(agg.y_range, (low, high), close=True)


@pytest.mark.parametrize('df', dfs)
def test_log_axis_points(df):
    axis = ds.core.LogAxis()
    logcoords = axis.compute_index(axis.compute_scale_and_translate((1, 10), 2), 2)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)

    sol = np.array([[5, 5], [5, 5]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, logcoords],
                       dims=['y', 'log_x'])
    assert_eq_xr(c_logx.points(df, 'log_x', 'y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, lincoords],
                       dims=['log_y', 'x'])
    assert_eq_xr(c_logy.points(df, 'x', 'log_y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, logcoords],
                       dims=['log_y', 'log_x'])
    assert_eq_xr(c_logxy.points(df, 'log_x', 'log_y', ds.count('i32')), out)


@pytest.mark.skipif(not sp, reason="spatialpandas not installed")
def test_points_geometry_point():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0., 2.), 3), 3)

    df = sp.GeoDataFrame({
        'geom': pd.array(
            [[0, 0], [0, 1], [1, 1], [0, 2], [1, 2], [2, 2]], dtype='Point[float64]'),
        'v': [1, 2, 2, 3, 3, 3]
    })

    cvs = ds.Canvas(plot_width=3, plot_height=3)
    agg = cvs.points(df, geometry='geom', agg=ds.sum('v'))
    sol = np.array([[1, nan, nan],
                    [2, 2,   nan],
                    [3, 3,   3]], dtype='float64')
    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 2), close=True)
    assert_eq_ndarray(agg.y_range, (0, 2), close=True)

    # Aggregation should not have triggered calculation of spatial index
    assert df.geom.array._sindex is None

    # Generate spatial index and check that we get the same result
    df.geom.array.sindex
    agg = cvs.points(df, geometry='geom', agg=ds.sum('v'))
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 2), close=True)
    assert_eq_ndarray(agg.y_range, (0, 2), close=True)


@pytest.mark.skipif(not sp, reason="spatialpandas not installed")
def test_points_geometry_multipoint():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0., 2.), 3), 3)

    df = sp.GeoDataFrame({
        'geom': pd.array(
            [[0, 0], [0, 1, 1, 1], [0, 2, 1, 2, 2, 2]], dtype='MultiPoint[float64]'),
        'v': [1, 2, 3]
    })

    cvs = ds.Canvas(plot_width=3, plot_height=3)
    agg = cvs.points(df, geometry='geom', agg=ds.sum('v'))
    sol = np.array([[1, nan, nan],
                    [2, 2,   nan],
                    [3, 3,   3]], dtype='float64')
    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 2), close=True)
    assert_eq_ndarray(agg.y_range, (0, 2), close=True)

    # Aggregation should not have triggered calculation of spatial index
    assert df.geom.array._sindex is None

    # Generate spatial index and check that we get the same result
    df.geom.array.sindex
    agg = cvs.points(df, geometry='geom', agg=ds.sum('v'))
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 2), close=True)
    assert_eq_ndarray(agg.y_range, (0, 2), close=True)


def test_line():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-3., 3.), 7), 7)

    df = pd.DataFrame({'x': [4, 0, -4, -3, -2, -1.9, 0, 10, 10, 0, 4],
                       'y': [0, -4, 0, 1, 2, 2.1, 4, 20, 30, 4, 0]})
    cvs = ds.Canvas(plot_width=7, plot_height=7,
                    x_range=(-3, 3), y_range=(-3, 3))
    agg = cvs.line(df, 'x', 'y', ds.count())
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


def test_points_on_edge():
    df = pd.DataFrame(dict(x=[0, 0.5, 1.1, 1.5, 2.2, 3, 3, 0],
                           y=[0, 0, 0, 0, 0, 0, 3, 3]))

    canvas = ds.Canvas(plot_width=3, plot_height=3,
                       x_range=(0, 3), y_range=(0, 3))

    agg = canvas.points(df, 'x', 'y', agg=ds.count())

    sol = np.array([[2, 2, 2],
                    [0, 0, 0],
                    [1, 0, 1]], dtype='int32')
    out = xr.DataArray(sol,
                       coords=[('x', [0.5, 1.5, 2.5]),
                               ('y', [0.5, 1.5, 2.5])],
                       dims=['y', 'x'])

    assert_eq_xr(agg, out)


def test_lines_on_edge():
    df = pd.DataFrame(dict(x=[0, 3, 3, 0],
                           y=[0, 0, 3, 3]))

    canvas = ds.Canvas(plot_width=3, plot_height=3,
                       x_range=(0, 3), y_range=(0, 3))

    agg = canvas.line(df, 'x', 'y', agg=ds.count())

    sol = np.array([[1, 1, 1],
                    [0, 0, 1],
                    [1, 1, 1]], dtype='int32')
    out = xr.DataArray(sol,
                       coords=[('x', [0.5, 1.5, 2.5]),
                               ('y', [0.5, 1.5, 2.5])],
                       dims=['y', 'x'])

    assert_eq_xr(agg, out)


@pytest.mark.parametrize('df', dfs)
def test_log_axis_line(df):
    axis = ds.core.LogAxis()
    logcoords = axis.compute_index(axis.compute_scale_and_translate((1, 10), 2), 2)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)

    sol = np.array([[4, 5], [5, 5]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, logcoords],
                       dims=['y', 'log_x'])
    assert_eq_xr(c_logx.line(df, 'log_x', 'y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, lincoords],
                       dims=['log_y', 'x'])
    assert_eq_xr(c_logy.line(df, 'x', 'log_y', ds.count('i32')), out)
    out = xr.DataArray(sol, coords=[logcoords, logcoords],
                       dims=['log_y', 'log_x'])
    assert_eq_xr(c_logxy.line(df, 'log_x', 'log_y', ds.count('i32')), out)


def test_subpixel_line_start():
    cvs = ds.Canvas(plot_width=5, plot_height=5, x_range=(1, 3), y_range=(0, 1))

    df = pd.DataFrame(dict(x=[1, 2, 3], y0=[0.0, 0.0, 0.0], y1=[0.0, 0.08, 0.04]))
    agg = cvs.line(df, 'x', ['y0', 'y1'], agg=ds.count(), axis=1)
    xcoords = axis.compute_index(axis.compute_scale_and_translate((1., 3), 5), 5)
    ycoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 5), 5)
    sol = np.array([
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype='i4')
    out = xr.DataArray(
        sol, coords=[ycoords, xcoords], dims=['y', 'x']
    )
    assert_eq_xr(agg, out)


def test_auto_range_line():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-10., 10.), 5), 5)

    df = pd.DataFrame({'x': [-10,  0, 10,   0, -10],
                       'y': [  0, 10,  0, -10,   0]})
    cvs = ds.Canvas(plot_width=5, plot_height=5)
    agg = cvs.line(df, 'x', 'y', ds.count())
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


@pytest.mark.skipif(not sp, reason="spatialpandas not installed")
@pytest.mark.parametrize('geom_data,geom_type', [
    ([0, 0, 1, 1, 2, 0, 0, 0], 'line'),
    ([[0, 0, 1, 1, 2, 0, 0, 0]], 'multiline'),
    ([0, 0, 1, 1, 2, 0, 0, 0], 'ring'),
    ([[0, 0, 1, 1, 2, 0, 0, 0]], 'polygon'),
    ([[[0, 0, 1, 1, 2, 0, 0, 0]]], 'multipolygon'),
])
def test_closed_ring_line(geom_data, geom_type):
    gdf = sp.GeoDataFrame(
        {'geometry': sp.GeoSeries([geom_data], dtype=geom_type)}
    )
    cvs = ds.Canvas(plot_width=4, plot_height=4)
    agg = cvs.line(gdf, geometry='geometry', agg=ds.count())

    coords_x = axis.compute_index(axis.compute_scale_and_translate((0., 2), 4), 4)
    coords_y = axis.compute_index(axis.compute_scale_and_translate((0., 1), 4), 4)
    sol = np.array([
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
    ])
    out = xr.DataArray(sol, coords=[coords_y, coords_x], dims=['y', 'x'])

    if geom_type.endswith("line"):
        # Closed rings represented as line/multiLine arrays will double count the
        # starting pixel
        out[0, 0] = 2

    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 2), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)


def test_trimesh_no_double_edge():
    """Assert that when two triangles share an edge that would normally get
    double-drawn, the edge is only drawn for the rightmost (or bottommost)
    triangle.
    """
    # Test left/right edge shared
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [1, 2]})
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

    # Test top/bottom edge shared
    verts = pd.DataFrame({'x': [3, 3, 1, 1, 3, 3],
                          'y': [4, 1, 4, 4, 5, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [3, 1]})
    # Plot dims and x/y ranges need to be set such that the edge is drawn twice:
    cvs = ds.Canvas(plot_width=22, plot_height=22, x_range=(0, 10), y_range=(0, 10))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud((agg.fillna(0) + 0.5).astype('i4').values)[10:20, :20], sol)

def test_trimesh_interp():
    """Assert that triangles are interpolated when vertex values are provided.
    """
    verts = pd.DataFrame({'x': [0, 5, 10],
                          'y': [0, 10, 0]})
    tris = pd.DataFrame({'v0': [0], 'v1': [1], 'v2': [2],
                         'val': [1]})
    cvs = ds.Canvas(plot_width=10, plot_height=10, x_range=(0, 10), y_range=(0, 10))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values), sol)

    verts = pd.DataFrame({'x': [0, 5, 10],
                          'y': [0, 10, 0],
                          'z': [1, 5, 3]})
    cvs = ds.Canvas(plot_width=10, plot_height=10, x_range=(0, 10), y_range=(0, 10))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 4, 0, 0, 0, 0],
        [0, 0, 0, 3, 3, 4, 4, 0, 0, 0],
        [0, 0, 0, 3, 3, 3, 3, 0, 0, 0],
        [0, 0, 2, 3, 3, 3, 3, 3, 0, 0],
        [0, 0, 2, 2, 2, 3, 3, 3, 0, 0],
        [0, 2, 2, 2, 2, 2, 3, 3, 3, 0],
        [0, 1, 1, 2, 2, 2, 2, 2, 3, 0],
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 3],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values), sol)

def test_trimesh_simplex_weights():
    """Assert that weighting the simplices works as expected.
    """
    # val is float
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [2., 4.]}) # floats
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)

    # val is int
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [3, 4]})
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)

def test_trimesh_vertex_weights():
    """Assert that weighting the vertices works as expected.
    """
    # z is float
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4],
                          'z': [1., 1., 1., 2., 2., 2.]}, columns=['x', 'y', 'z'])
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5]})
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='f8')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0.).values)[:5], sol)

    # val is int
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4],
                          'val': [2, 2, 2, 3, 3, 3]}, columns=['x', 'y', 'val'])
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5]})
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 3, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)

def test_trimesh_winding_detect():
    """Assert that CCW windings get converted to CW.
    """
    # val is int, winding is CCW
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [2, 5], 'v2': [1, 4], 'val': [3, 4]})
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)

    # val is float, winding is CCW
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [2, 5], 'v2': [1, 4], 'val': [3., 4.]}) # floats
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)

def test_trimesh_mesharg():
    """Assert that the ``mesh`` argument results in the same rasterization,
    despite the ``vertices`` and ``simplices`` arguments changing.
    """
    # z is float
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4],
                          'y': [4, 5, 5, 5, 4, 4],
                          'z': [1., 1., 1., 2., 2., 2.]}, columns=['x', 'y', 'z'])
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5]})
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='f8')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0.).values)[:5], sol)

    mesh = ds.utils.mesh(verts, tris)
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts[:1], tris[:1], mesh=mesh)
    np.testing.assert_array_equal(np.flipud(agg.fillna(0.).values)[:5], sol)

def test_trimesh_agg_api():
    """Assert that the trimesh aggregation API properly handles weights on the simplices."""
    pts = pd.DataFrame({'x': [1, 3, 4, 3, 3],
                        'y': [2, 1, 2, 1, 4]},
                       columns=['x', 'y'])
    tris = pd.DataFrame({'n1': [4, 1],
                         'n2': [1, 4],
                         'n3': [2, 0],
                         'weight': [0.83231525, 1.3053126]},
                        columns=['n1', 'n2', 'n3', 'weight'])
    cvs = ds.Canvas(x_range=(0, 10), y_range=(0, 10))
    agg = cvs.trimesh(pts, tris, agg=ds.mean('weight'))
    assert agg.shape == (600, 600)

    assert_eq_ndarray(agg.x_range, (0, 10), close=True)
    assert_eq_ndarray(agg.y_range, (0, 10), close=True)


def test_bug_570():
    # See https://github.com/holoviz/datashader/issues/570
    df = pd.DataFrame({
        'Time': [1456353642.2053893, 1456353642.2917893],
        'data': [-59.4948743433377, 506.4847376716022],
    }, columns=['Time', 'data'])

    x_range = (1456323293.9859753, 1456374687.0009754)
    y_range = (-228.56721300380943, 460.4042291124646)

    cvs = ds.Canvas(x_range=x_range, y_range=y_range,
                    plot_height=300, plot_width=1000)
    agg = cvs.line(df, 'Time', 'data', agg=ds.count())

    # Check location of line
    yi, xi = np.where(agg.values == 1)
    assert np.array_equal(yi, np.arange(73, 300))
    assert np.array_equal(xi, np.array([590] * len(yi)))


# # Line tests
line_manual_range_params = [
    # axis1 none constant
    ([{
        'x0': [4, -4],
        'x1': [0,  0],
        'x2': [-4, 4],
        'y0': [0,  0],
        'y1': [-4, 4],
        'y2': [0,  0]
    }], dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis1 x constant
    ([{
        'y0': [0,  0],
        'y1': [-4, 4],
        'y2': [0,  0]
    }], dict(x=np.array([-4, 0, 4]), y=['y0', 'y1', 'y2'], axis=1)),

    # axis1 y constant
    ([{
        'x0': [0, 0],
        'x1': [-4, 4],
        'x2': [0, 0]
    }], dict(x=['x0', 'x1', 'x2'], y=np.array([-4, 0, 4]), axis=1)),

    # axis0 single
    ([{
        'x': [0, -4, 0, np.nan, 0,  4, 0],
        'y': [-4, 0, 4, np.nan, -4, 0, 4],
    }], dict(x='x', y='y', axis=0)),

    # axis0 multi
    ([{
        'x0': [0, -4, 0],
        'x1': [0,  4, 0],
        'y0': [-4, 0, 4],
        'y1': [-4, 0, 4],
    }], dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)),

    # axis0 multi with string
    ([{
        'x0': [0, -4, 0],
        'x1': [0,  4, 0],
        'y0': [-4, 0, 4],
        'y1': [-4, 0, 4],
    }], dict(x=['x0', 'x1'], y='y0', axis=0)),

    # axis1 ragged arrays
    ([{
        'x': pd.array([[4, 0], [0, -4, 0, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4], [-4, 0, 4, 0]], dtype='Ragged[float32]')
    }], dict(x='x', y='y', axis=1)),
]
if sp:
    line_manual_range_params.append(
        # geometry
        ([{
            'geom': pd.array(
                [[4, 0, 0, -4], [0, -4, -4, 0, 0, 4, 4, 0]], dtype='Line[float32]'
            ),
        }], dict(geometry='geom'))
    )
@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_args,cvs_kwargs', line_manual_range_params)
def test_line_manual_range(DataFrame, df_args, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if (isinstance(getattr(df_args[0].get('x', []), 'dtype', ''), RaggedDtype) or
                sp and isinstance(
                    getattr(df_args[0].get('geom', []), 'dtype', ''), LineDtype
                )
        ):
            pytest.skip("cudf DataFrames do not support extension types")

    df = DataFrame(geo='geometry' in cvs_kwargs, *df_args)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(
        axis.compute_scale_and_translate((-3., 3.), 7), 7)

    cvs = ds.Canvas(plot_width=7, plot_height=7,
                    x_range=(-3, 3), y_range=(-3, 3))

    agg = cvs.line(df, agg=ds.count(), **cvs_kwargs)

    sol = np.array([[0, 0, 1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
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
    ([{
        'x0': [0,  0],
        'x1': [-4, 4],
        'x2': [0,  0],
        'y0': [-4, -4],
        'y1': [0,  0],
        'y2': [4,  4]
    }], dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

    # axis1 y constant
    ([{
        'x0': [0, 0],
        'x1': [-4, 4],
        'x2': [0, 0]
    }], dict(x=['x0', 'x1', 'x2'], y=np.array([-4, 0, 4]), axis=1)),

    # axis0 single
    ([{
        'x': [0, -4, 0, np.nan, 0,  4, 0],
        'y': [-4, 0, 4, np.nan, -4, 0, 4],
    }], dict(x='x', y='y', axis=0)),

    # axis0 multi
    ([{
        'x0': [0, -4, 0],
        'x1': [0,  4, 0],
        'y0': [-4, 0, 4],
        'y1': [-4, 0, 4],
    }], dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)),

    # axis0 multi with string
    ([{
        'x0': [0, -4, 0],
        'x1': [0,  4, 0],
        'y0': [-4, 0, 4],
        'y1': [-4, 0, 4],
    }], dict(x=['x0', 'x1'], y='y0', axis=0)),

    # axis1 ragged arrays
    ([{
        'x': pd.array([[0, -4, 0], [0,  4, 0]], dtype='Ragged[float32]'),
        'y': pd.array([[-4, 0, 4], [-4, 0, 4]], dtype='Ragged[float32]')
    }], dict(x='x', y='y', axis=1)),
]
if sp:
    line_autorange_params.append(
        # geometry
        ([{
            'geom': pd.array(
                [[0, -4, -4, 0, 0, 4], [0, -4,  4, 0, 0, 4]], dtype='Line[float32]'
            ),
        }], dict(geometry='geom'))
    )
@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_args,cvs_kwargs', line_autorange_params)
@pytest.mark.parametrize('line_width', [0, 1])
def test_line_autorange(DataFrame, df_args, cvs_kwargs, line_width):
    if cudf and DataFrame is cudf_DataFrame:
        if (isinstance(getattr(df_args[0].get('x', []), 'dtype', ''), RaggedDtype) or
                sp and isinstance(
                    getattr(df_args[0].get('geom', []), 'dtype', ''), LineDtype
                )
        ):
            pytest.skip("cudf DataFrames do not support extension types")

        if line_width > 0:
            pytest.skip("cudf DataFrames do not support antialiased lines")

    df = DataFrame(geo='geometry' in cvs_kwargs, *df_args)

    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 9), 9)

    cvs = ds.Canvas(plot_width=9, plot_height=9)

    agg = cvs.line(df, agg=ds.count(), line_width=line_width, **cvs_kwargs)

    if line_width > 0:
        sol = np.array([
            [np.nan,   np.nan,   np.nan,   0.646447, 1.292893, 0.646447, np.nan,   np.nan,   np.nan  ],
            [np.nan,   np.nan,   0.646447, 0.646447, np.nan,   0.646447, 0.646447, np.nan,   np.nan  ],
            [np.nan,   0.646447, 0.646447, np.nan,   np.nan,   np.nan,   0.646447, 0.646447, np.nan  ],
            [0.646447, 0.646447, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   0.646447, 0.646447],
            [0.646447, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   0.646447],
            [0.646447, 0.646447, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   0.646447, 0.646447],
            [np.nan,   0.646447, 0.646447, np.nan,   np.nan,   np.nan,   0.646447, 0.646447, np.nan  ],
            [np.nan,   np.nan,   0.646447, 0.646447, np.nan,   0.646447, 0.646447, np.nan,   np.nan  ],
            [np.nan,   np.nan,   np.nan,   0.646447, 1.292893, 0.646447, np.nan,   np.nan,   np.nan  ]
        ], dtype='f4')
    else:
        sol = np.array([[0, 0, 0, 0, 2, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 1, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 2, 0, 0, 0, 0]], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out, close=(line_width > 0))
    assert_eq_ndarray(agg.x_range, (-4, 4), close=True)
    assert_eq_ndarray(agg.y_range, (-4, 4), close=True)


@pytest.mark.parametrize('DataFrame', DataFrames)
def test_line_autorange_axis1_x_constant(DataFrame):
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 9), 9)

    xs = np.array([-4, 0, 4])
    df = DataFrame({
        'y0': [0,  0],
        'y1': [-4, 4],
        'y2': [0,  0]
    })

    cvs = ds.Canvas(plot_width=9, plot_height=9)

    agg = cvs.line(df,
                   xs,
                   ['y0', 'y1', 'y2'],
                   ds.count(),
                   axis=1)

    sol = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [2, 0, 0, 0, 0, 0, 0, 0, 2],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


# Sum aggregate
@pytest.mark.parametrize('DataFrame', DataFrames)
def test_line_agg_sum_axis1_none_constant(DataFrame):
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-3., 3.), 7), 7)

    df = DataFrame({
        'x0': [4, -4],
        'x1': [0,  0],
        'x2': [-4, 4],
        'y0': [0,  0],
        'y1': [-4, 4],
        'y2': [0,  0],
        'v': [7, 9]
    })

    cvs = ds.Canvas(plot_width=7, plot_height=7,
                    x_range=(-3, 3), y_range=(-3, 3))

    agg = cvs.line(df,
                   ['x0', 'x1', 'x2'],
                   ['y0', 'y1', 'y2'],
                   ds.sum('v'),
                   axis=1)
    nan = np.nan
    sol = np.array([[nan, nan, 7,   nan, 7,   nan, nan],
                    [nan, 7,   nan, nan, nan, 7,   nan],
                    [7,   nan, nan, nan, nan, nan, 7],
                    [nan, nan, nan, nan, nan, nan, nan],
                    [9,   nan, nan, nan, nan, nan, 9],
                    [nan, 9,   nan, nan, nan, 9,   nan],
                    [nan, nan, 9,   nan, 9,   nan, nan]], dtype='float32')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)


def test_line_autorange_axis1_ragged():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 9), 9)

    df = pd.DataFrame({
        'x': pd.array([[4, 0], [0, -4, 0, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4], [-4, 0, 4, 0]], dtype='Ragged[float32]')
    })

    cvs = ds.Canvas(plot_width=9, plot_height=9)

    agg = cvs.line(df,
                   'x',
                   'y',
                   ds.count(),
                   axis=1)

    sol = np.array([[0, 0, 0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 2],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords, lincoords],
                       dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (-4, 4), close=True)
    assert_eq_ndarray(agg.y_range, (-4, 4), close=True)


@pytest.mark.parametrize('DataFrame', DataFrames)
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
        'x0': [-4., -2., 0],
        'x1': [np.nan, 2, 4],
        'y0': [0, -4, 0],
        'y1': [np.nan, 4, 0],
    }, dtype='float32'), dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': pd.array([[-4, -2, 0], [2, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4, 0], [4, 0]], dtype='Ragged[float32]')
    }), dict(x='x', y='y', axis=1))
])
def test_area_to_zero_fixedrange(DataFrame, df_kwargs, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if isinstance(getattr(df_kwargs['data'].get('x', []), 'dtype', ''), RaggedDtype):
            pytest.skip("cudf DataFrames do not support extension types")

    df = DataFrame(**df_kwargs)

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-2.25, 2.25), 5), 5)

    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-3.75, 3.75), 9), 9)

    cvs = ds.Canvas(plot_width=9, plot_height=5,
                    x_range=[-3.75, 3.75], y_range=[-2.25, 2.25])

    agg = cvs.area(df, agg=ds.count(), **cvs_kwargs)

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


@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [
    # axis1 none constant
    (dict(data={
        'x0': [-4, 0],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, 0],
        'y1': [-4, -4],
        'y2': [0, 0]
    }, dtype='float32'), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)),

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
        'x': pd.array([[-4, -2, 0], [0, 2, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4, 0], [0, -4, 0]], dtype='Ragged[float32]')
    }), dict(x='x', y='y', axis=1))
])
def test_area_to_zero_autorange(DataFrame, df_kwargs, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if isinstance(getattr(df_kwargs['data'].get('x', []), 'dtype', ''), RaggedDtype):
            pytest.skip("cudf DataFrames do not support extension types")

    df = DataFrame(**df_kwargs)

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 0.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    agg = cvs.area(df, agg=ds.count(), **cvs_kwargs)

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


@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [
    # axis1 none constant
    (dict(data={
        'x0': [-4, np.nan],
        # 'x0': [-4, 0],
        'x1': [-2, 2],
        'x2': [0, 4],
        'y0': [0, np.nan],
        # 'y0': [0, 1],
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
        'x': pd.array([[-4, -2, 0], [2, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4, 0], [4, 0]], dtype='Ragged[float32]')
    }), dict(x='x', y='y', axis=1))
])
def test_area_to_zero_autorange_gap(DataFrame, df_kwargs, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if isinstance(getattr(df_kwargs['data'].get('x', []), 'dtype', ''), RaggedDtype):
            pytest.skip("cudf DataFrames do not support extension types")

    df = DataFrame(**df_kwargs)

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    agg = cvs.area(df, agg=ds.count(), **cvs_kwargs)

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
    assert_eq_ndarray(agg.x_range, (-4, 4), close=True)
    assert_eq_ndarray(agg.y_range, (-4, 4), close=True)


@pytest.mark.parametrize('DataFrame', DataFrames)
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
    }, dtype='float32'), dict(x=['x0', 'x1'], y=['y0', 'y1'],
                              y_stack=['y2', 'y3'], axis=0)),

    # axis0 multi, y string
    (dict(data={
        'x0': [-4, -2, 0],
        'x1': [0, 2, 4],
        'y0': [0, -4, 0],
        'y2': [0, -2, 0],
    }, dtype='float32'), dict(x=['x0', 'x1'], y='y0', y_stack='y2', axis=0)),

    # axis1 ragged arrays
    (dict(data={
        'x': pd.array([[-4, -2, 0], [0, 2, 4]], dtype='Ragged[float32]'),
        'y': pd.array([[0, -4, 0], [0, -4, 0]], dtype='Ragged[float32]'),
        'y_stack': pd.array([[0, -2, 0], [0, -2, 0]], dtype='Ragged[float32]')
    }), dict(x='x', y='y', y_stack='y_stack', axis=1))
])
def test_area_to_line_autorange(DataFrame, df_kwargs, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if isinstance(getattr(df_kwargs['data'].get('x', []), 'dtype', ''), RaggedDtype):
            pytest.skip("cudf DataFrames do not support extension types")

    df = DataFrame(**df_kwargs)

    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 0.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    agg = cvs.area(df, agg=ds.count(), **cvs_kwargs)

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


def test_area_to_line_autorange_gap():
    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 7), 7)
    lincoords_x = axis.compute_index(
        axis.compute_scale_and_translate((-4., 4.), 13), 13)

    cvs = ds.Canvas(plot_width=13, plot_height=7)

    df = pd.DataFrame({
        'x': [-4, -2, 0, np.nan, 2, 4],
        'y0': [0, -4, 0, np.nan, 4, 0],
        'y1': [0,  0, 0, np.nan, 0, 0],
    })

    # When a line is specified to fill to, this line is not included in
    # the fill.  So we expect the y=0 line to not be filled.
    agg = cvs.area(df, 'x', 'y0', ds.count(), y_stack='y1')

    sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
                   dtype='i4')

    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x],
                       dims=['y0', 'x'])
    assert_eq_xr(agg, out)


# Using local versions of nan-aware combinations rather than those in
# utils.py.  These versions are not always applicable, e.g. if summming
# a positive and negative value to total exactly zero will be wrong here.
def nanmax(arr0, arr1):
    mask = np.logical_and(np.isnan(arr0), np.isnan(arr1))
    ret = np.maximum(np.nan_to_num(arr0, nan=0.0), np.nan_to_num(arr1, nan=0.0))
    ret[mask] = np.nan
    return ret

def nanmin(arr0, arr1):
    mask = np.logical_and(np.isnan(arr0), np.isnan(arr1))
    ret = np.minimum(np.nan_to_num(arr0, nan=1e10), np.nan_to_num(arr1, nan=1e10))
    ret[mask] = np.nan
    return ret

def nansum(arr0, arr1):
    mask = np.logical_and(np.isnan(arr0), np.isnan(arr1))
    ret = np.nan_to_num(arr0, nan=0.0) + np.nan_to_num(arr1, nan=0.0)
    ret[mask] = np.nan
    return ret

def rowmax(arr0, arr1):
    return np.maximum(arr0, arr1)

def rowmin(arr0, arr1):
    bigint = np.max([np.max(arr0), np.max(arr1)]) + 1
    arr0[arr0 < 0] = bigint
    arr1[arr1 < 0] = bigint
    ret = np.minimum(arr0, arr1)
    ret[ret == bigint] = -1
    return ret

line_antialias_df = pd.DataFrame(dict(
    # Self-intersecting line.
    x0=np.asarray([0, 1, 1, 0]),
    y0=np.asarray([0, 1, 0, 1]),
    # Non-self-intersecting line.
    x1=np.linspace(0.0, 1.0, 4),
    y1=np.linspace(0.125, 0.2, 4),
    value=[3, 3, 3, 3],
))
line_antialias_sol_0 = np.array([
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, np.nan],
    [np.nan, 1.0,      0.292893, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   0.292893, 1.0,    np.nan],
    [np.nan, 0.292893, 1.0,      0.292893, np.nan,   np.nan,   np.nan,   0.292893, 1.0,      1.0,    np.nan],
    [np.nan, np.nan,   0.292893, 1.0,      0.292893, np.nan,   0.292893, 1.0,      0.292893, 1.0,    np.nan],
    [np.nan, np.nan,   np.nan,   0.292893, 1.0,      0.292893, 1.0,      0.292893, np.nan,   1.0,    np.nan],
    [np.nan, np.nan,   np.nan,   np.nan,   0.292893, 1.0,      0.292893, np.nan,   np.nan,   1.0,    np.nan],
    [np.nan, np.nan,   np.nan,   0.292893, 1.0,      0.292893, 1.0,      0.292893, np.nan,   1.0,    np.nan],
    [np.nan, np.nan,   0.292893, 1.0,      0.292893, np.nan,   0.292893, 1.0,      0.292893, 1.0,    np.nan],
    [np.nan, 0.292893, 1.0,      0.292893, np.nan,   np.nan,   np.nan,   0.292893, 1.0,      1.0,    np.nan],
    [np.nan, 1.0,      0.292893, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   0.292893, 1.0,    np.nan],
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, np.nan],
])
line_antialias_sol_0_intersect = np.array([
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, np.nan],
    [np.nan, 1.0,      0.292893, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   0.292893, 1.0,    np.nan],
    [np.nan, 0.292893, 1.0,      0.292893, np.nan,   np.nan,   np.nan,   0.292893, 1.0,      1.0,    np.nan],
    [np.nan, np.nan,   0.292893, 1.0,      0.292893, np.nan,   0.292893, 1.0,      0.292893, 1.0,    np.nan],
    [np.nan, np.nan,   np.nan,   0.292893, 1.0,      0.585786, 1.0,      0.292893, np.nan,   1.0,    np.nan],
    [np.nan, np.nan,   np.nan,   np.nan,   0.585786, 2.0,      0.585786, np.nan,   np.nan,   1.0,    np.nan],
    [np.nan, np.nan,   np.nan,   0.292893, 1.0,      0.585786, 1.0,      0.292893, np.nan,   1.0,    np.nan],
    [np.nan, np.nan,   0.292893, 1.0,      0.292893, np.nan,   0.292893, 1.0,      0.292893, 1.0,    np.nan],
    [np.nan, 0.292893, 1.0,      0.292893, np.nan,   np.nan,   np.nan,   0.292893, 1.0,      1.0,    np.nan],
    [np.nan, 1.0,      0.292893, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   0.292893, 1.0,    np.nan],
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, np.nan],
])
line_antialias_sol_1 = np.array([
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,  np.nan,  np.nan,   np.nan,  np.nan,  np.nan],
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,  np.nan,  np.nan,   np.nan,  np.nan,  np.nan],
    [np.nan, 1.0,      0.92521,  0.85042,  0.77563,  0.70084, 0.62605, 0.55126 , 0.47647, 0.40168, np.nan],
    [np.nan, 0.002801, 0.077591, 0.152381, 0.227171, 0.30196, 0.37675, 0.45154 , 0.52633, 0.6,     np.nan],
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,  np.nan,  np.nan,   np.nan,  np.nan,  np.nan],
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,  np.nan,  np.nan,   np.nan,  np.nan,  np.nan],
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,  np.nan,  np.nan,   np.nan,  np.nan,  np.nan],
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,  np.nan,  np.nan,   np.nan,  np.nan,  np.nan],
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,  np.nan,  np.nan,   np.nan,  np.nan,  np.nan],
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,  np.nan,  np.nan,   np.nan,  np.nan,  np.nan],
    [np.nan, np.nan,   np.nan,   np.nan,   np.nan,   np.nan,  np.nan,  np.nan,   np.nan,  np.nan,  np.nan],
])
line_antialias_sol_min_index_0 = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1,  0,  0, -1, -1, -1, -1, -1,  2,  1, -1],
    [-1,  0,  0,  0, -1, -1, -1,  2,  2,  1, -1],
    [-1, -1,  0,  0,  0, -1,  2,  2,  2,  1, -1],
    [-1, -1, -1,  0,  0,  0,  2,  2, -1,  1, -1],
    [-1, -1, -1, -1,  0,  0,  0, -1, -1,  1, -1],
    [-1, -1, -1,  2,  2,  0,  0,  0, -1,  1, -1],
    [-1, -1,  2,  2,  2, -1,  0,  0,  0,  1, -1],
    [-1,  2,  2,  2, -1, -1, -1,  0,  0,  0, -1],
    [-1,  2,  2, -1, -1, -1, -1, -1,  0,  0, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
], dtype=np.int64)
line_antialias_sol_max_index_0 = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1,  0,  0, -1, -1, -1, -1, -1,  2,  2, -1],
    [-1,  0,  0,  0, -1, -1, -1,  2,  2,  2, -1],
    [-1, -1,  0,  0,  0, -1,  2,  2,  2,  1, -1],
    [-1, -1, -1,  0,  0,  2,  2,  2, -1,  1, -1],
    [-1, -1, -1, -1,  2,  2,  2, -1, -1,  1, -1],
    [-1, -1, -1,  2,  2,  2,  0,  0, -1,  1, -1],
    [-1, -1,  2,  2,  2, -1,  0,  0,  0,  1, -1],
    [-1,  2,  2,  2, -1, -1, -1,  0,  0,  1, -1],
    [-1,  2,  2, -1, -1, -1, -1, -1,  0,  1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
], dtype=np.int64)
line_antialias_sol_min_index_1 = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1,  0,  0,  0,  0,  1,  1,  1,  2,  2, -1],
    [-1,  0,  0,  0,  0,  1,  1,  1,  2,  2, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
], dtype=np.int64)
line_antialias_sol_max_index_1 = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1,  0,  0,  1,  1,  1,  2,  2,  2,  2, -1],
    [-1,  0,  0,  0,  1,  1,  2,  2,  2,  2, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
], dtype=np.int64)

def test_line_antialias():
    x_range = y_range = (-0.1875, 1.1875)
    cvs = ds.Canvas(plot_width=11, plot_height=11, x_range=x_range, y_range=y_range)

    # First line only, self-intersects
    kwargs = dict(source=line_antialias_df, x="x0", y="y0", line_width=1)
    agg = cvs.line(agg=ds.any(), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_0, close=True)

    agg = cvs.line(agg=ds.count(self_intersect=False), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_0, close=True)

    agg = cvs.line(agg=ds.count(self_intersect=True), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_0_intersect, close=True)

    agg = cvs.line(agg=ds.sum("value", self_intersect=False), **kwargs)
    assert_eq_ndarray(agg.data, 3*line_antialias_sol_0, close=True)

    agg = cvs.line(agg=ds.sum("value", self_intersect=True), **kwargs)
    assert_eq_ndarray(agg.data, 3*line_antialias_sol_0_intersect, close=True)

    agg = cvs.line(agg=ds.max("value"), **kwargs)
    assert_eq_ndarray(agg.data, 3*line_antialias_sol_0, close=True)

    agg = cvs.line(agg=ds.min("value"), **kwargs)
    assert_eq_ndarray(agg.data, 3*line_antialias_sol_0, close=True)

    agg = cvs.line(agg=ds.first("value"), **kwargs)
    assert_eq_ndarray(agg.data, 3*line_antialias_sol_0, close=True)

    agg = cvs.line(agg=ds.last("value"), **kwargs)
    assert_eq_ndarray(agg.data, 3*line_antialias_sol_0, close=True)

    agg = cvs.line(agg=ds.mean("value"), **kwargs)
    # Sum = 3*count so mean is 3 everywhere that there is any fraction of an antialiased line
    sol = np.where(line_antialias_sol_0 > 0, 3.0, np.nan)
    assert_eq_ndarray(agg.data, sol, close=True)

    agg = cvs.line(agg=ds._min_row_index(), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_min_index_0)

    agg = cvs.line(agg=ds._max_row_index(), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_max_index_0)

    agg = cvs.line(agg=ds._min_n_row_index(n=2), **kwargs)
    assert_eq_ndarray(agg[:, :, 0].data, line_antialias_sol_min_index_0)
    sol = np.full((11, 11), -1)
    sol[(4, 5, 5, 5, 6, 1, 2), (5, 4, 5, 6, 5, 9, 9)] = 2
    sol[8:10, 9] = 1
    assert_eq_ndarray(agg[:, :, 1].data, sol)

    agg = cvs.line(agg=ds._max_n_row_index(n=2), **kwargs)
    assert_eq_ndarray(agg[:, :, 0].data, line_antialias_sol_max_index_0)
    sol = np.full((11, 11), -1)
    sol[(4, 5, 5, 5, 6, 8, 9), (5, 4, 5, 6, 5, 9, 9)] = 0
    sol[1:3, 9] = 1
    assert_eq_ndarray(agg[:, :, 1].data, sol)

    agg = cvs.line(agg=ds.max_n("value", n=2), **kwargs)
    assert_eq_ndarray(agg[:, :, 0].data, 3*line_antialias_sol_0, close=True)
    sol = np.full((11, 11), np.nan)
    sol[(1, 5, 9), (9, 5, 9)] = 3.0
    sol[(2, 4, 5, 5, 6, 8), (9, 5, 4, 6, 5, 9)] = 0.878680
    assert_eq_ndarray(agg[:, :, 1].data, sol, close=True)

    agg = cvs.line(agg=ds.min_n("value", n=2), **kwargs)
    sol = 3*line_antialias_sol_0
    sol[(2, 8), (9, 9)] = 0.878680
    assert_eq_ndarray(agg[:, :, 0].data, sol, close=True)
    sol = np.full((11, 11), np.nan)
    sol[(1, 2, 5, 8, 9), (9, 9, 5, 9, 9)] = 3.0
    sol[(4, 5, 5, 6), (5, 4, 6, 5)] = 0.878680
    assert_eq_ndarray(agg[:, :, 1].data, sol, close=True)

    agg = cvs.line(agg=ds.first_n("value", n=2), **kwargs)
    sol = 3*line_antialias_sol_0
    sol[8, 9] = 0.878680
    assert_eq_ndarray(agg[:, :, 0].data, sol, close=True)
    sol = np.full((11, 11), np.nan)
    sol[(1, 5, 8, 9), (9, 5, 9, 9)] = 3.0
    sol[(2, 4, 5, 5, 6), (9, 5, 4, 6, 5)] = 0.878680
    assert_eq_ndarray(agg[:, :, 1].data, sol, close=True)

    # Second line only, doesn't self-intersect
    kwargs = dict(source=line_antialias_df, x="x1", y="y1", line_width=1)
    agg = cvs.line(agg=ds.any(), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_1, close=True)

    agg = cvs.line(agg=ds.count(self_intersect=False), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_1, close=True)

    agg = cvs.line(agg=ds.count(self_intersect=True), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_1, close=True)

    agg = cvs.line(agg=ds.sum("value", self_intersect=False), **kwargs)
    assert_eq_ndarray(agg.data, 3*line_antialias_sol_1, close=True)

    agg = cvs.line(agg=ds.sum("value", self_intersect=True), **kwargs)
    assert_eq_ndarray(agg.data, 3*line_antialias_sol_1, close=True)

    agg = cvs.line(agg=ds.max("value"), **kwargs)
    assert_eq_ndarray(agg.data, 3*line_antialias_sol_1, close=True)

    agg = cvs.line(agg=ds.min("value"), **kwargs)
    assert_eq_ndarray(agg.data, 3*line_antialias_sol_1, close=True)

    agg = cvs.line(agg=ds.first("value"), **kwargs)
    assert_eq_ndarray(agg.data, 3*line_antialias_sol_1, close=True)

    agg = cvs.line(agg=ds.last("value"), **kwargs)
    assert_eq_ndarray(agg.data, 3*line_antialias_sol_1, close=True)

    agg = cvs.line(agg=ds.mean("value"), **kwargs)
    sol_mean = np.where(line_antialias_sol_1 > 0, 3.0, np.nan)
    assert_eq_ndarray(agg.data, sol_mean, close=True)

    agg = cvs.line(agg=ds._min_row_index(), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_min_index_1)

    agg = cvs.line(agg=ds._max_row_index(), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_max_index_1)

    agg = cvs.line(agg=ds._min_n_row_index(n=2), **kwargs)
    assert_eq_ndarray(agg[:, :, 0].data, line_antialias_sol_min_index_1)
    sol = np.full((11, 11), -1)
    sol[(2, 2, 3), (3, 4, 4)] = 1
    sol[2:4, 6:8] = 2
    assert_eq_ndarray(agg[:, :, 1].data, sol)

    agg = cvs.line(agg=ds._max_n_row_index(n=2), **kwargs)
    assert_eq_ndarray(agg[:, :, 0].data, line_antialias_sol_max_index_1)
    sol = np.full((11, 11), -1)
    sol[(2, 2, 3), (3, 4, 4)] = 0
    sol[2:4, 6:8] = 1
    assert_eq_ndarray(agg[:, :, 1].data, sol)

    agg = cvs.line(agg=ds.max_n("value", n=2), **kwargs)
    assert_eq_ndarray(agg[:, :, 0].data, 3*line_antialias_sol_1, close=True)
    sol = np.full((11, 11), np.nan)
    sol[2, 3:8] = (0.911939, 1.833810, nan, 1.437950, 0.667619)
    sol[(3, 3, 3), (4, 6, 7)] = (0.4, 0.940874, 0.309275)
    assert_eq_ndarray(agg[:, :, 1].data, sol, close=True)

    agg = cvs.line(agg=ds.min_n("value", n=2), **kwargs)
    sol = np.full((11, 11), np.nan)
    sol[2, 1:-1] = [3.0, 2.775630, 0.911939, 1.833810, 2.1025216, 1.437950, 0.667619, 1.429411, 1.205041]
    sol[3, 1:-1] = [0.008402, 0.232772, 0.457142, 0.4, 0.905881, 0.940874, 0.309275, 1.578991, 1.8]
    assert_eq_ndarray(agg[:, :, 0].data, sol, close=True)
    sol = np.full((11, 11), np.nan)
    sol[2, 3:8] = (2.551260, 2.326890, nan, 1.878151, 1.653781)
    sol[(3, 3, 3), (4, 6, 7)] = (0.681512, 1.130251, 1.354621)
    assert_eq_ndarray(agg[:, :, 1].data, sol, close=True)

    agg = cvs.line(agg=ds.first_n("value", n=2), **kwargs)
    sol = 3*line_antialias_sol_1
    sol[(2, 2, 3, 3), (4, 7, 4, 7)] = (1.833810, 0.667619, 0.4, 0.309275)
    assert_eq_ndarray(agg[:, :, 0].data, sol, close=True)
    sol = np.full((11, 11), np.nan)
    sol[2, 3:8] = (0.911939, 2.326890, nan, 1.437950, 1.653781)
    sol[(3, 3, 3), (4, 6, 7)] = (0.681512, 0.940874, 1.354621)
    assert_eq_ndarray(agg[:, :, 1].data, sol, close=True)

    agg = cvs.line(agg=ds.last_n("value", n=2), **kwargs)
    sol = 3*line_antialias_sol_1
    sol[(2, 2, 3), (3, 6, 6)] = (0.911939, 1.437950, 0.940874)
    assert_eq_ndarray(agg[:, :, 0].data, sol, close=True)

    # Both lines.
    kwargs = dict(source=line_antialias_df, x=["x0", "x1"], y=["y0", "y1"], line_width=1)
    agg = cvs.line(agg=ds.any(), **kwargs)
    sol_max = nanmax(line_antialias_sol_0, line_antialias_sol_1)
    assert_eq_ndarray(agg.data, sol_max, close=True)

    agg = cvs.line(agg=ds.count(self_intersect=False), **kwargs)
    sol_count = nansum(line_antialias_sol_0, line_antialias_sol_1)
    assert_eq_ndarray(agg.data, sol_count, close=True)

    agg = cvs.line(agg=ds.count(self_intersect=True), **kwargs)
    sol_count_intersect = nansum(line_antialias_sol_0_intersect, line_antialias_sol_1)
    assert_eq_ndarray(agg.data, sol_count_intersect, close=True)

    agg = cvs.line(agg=ds.sum("value", self_intersect=False), **kwargs)
    assert_eq_ndarray(agg.data, 3*sol_count, close=True)

    agg = cvs.line(agg=ds.sum("value", self_intersect=True), **kwargs)
    assert_eq_ndarray(agg.data, 3*sol_count_intersect, close=True)

    agg = cvs.line(agg=ds.max("value"), **kwargs)
    assert_eq_ndarray(agg.data, 3*sol_max, close=True)

    agg = cvs.line(agg=ds.min("value"), **kwargs)
    sol_min = nanmin(line_antialias_sol_0, line_antialias_sol_1)
    assert_eq_ndarray(agg.data, 3*sol_min, close=True)

    agg = cvs.line(agg=ds.first("value"), **kwargs)
    sol_first = 3*np.where(np.isnan(line_antialias_sol_0), line_antialias_sol_1, line_antialias_sol_0)
    assert_eq_ndarray(agg.data, sol_first, close=True)

    agg = cvs.line(agg=ds.last("value"), **kwargs)
    sol_last = 3*np.where(np.isnan(line_antialias_sol_1), line_antialias_sol_0, line_antialias_sol_1)
    assert_eq_ndarray(agg.data, sol_last, close=True)

    agg = cvs.line(agg=ds.mean("value"), **kwargs)
    sol_mean = np.where(sol_count>0, 3.0, np.nan)
    assert_eq_ndarray(agg.data, sol_mean, close=True)

    agg = cvs.line(agg=ds._min_row_index(), **kwargs)
    sol_min_row = rowmin(line_antialias_sol_min_index_0, line_antialias_sol_min_index_1)
    assert_eq_ndarray(agg.data, sol_min_row)

    agg = cvs.line(agg=ds._max_row_index(), **kwargs)
    sol_max_row = rowmax(line_antialias_sol_max_index_0, line_antialias_sol_max_index_1)
    assert_eq_ndarray(agg.data, sol_max_row)

    agg = cvs.line(agg=ds._min_n_row_index(n=2), **kwargs)
    assert_eq_ndarray(agg.data[:, :, 0], sol_min_row)

    agg = cvs.line(agg=ds._max_n_row_index(n=2), **kwargs)
    assert_eq_ndarray(agg.data[:, :, 0], sol_max_row)

    assert_eq_ndarray(agg.x_range, x_range, close=True)
    assert_eq_ndarray(agg.y_range, y_range, close=True)


def test_line_antialias_summary():
    kwargs = dict(source=line_antialias_df, x=["x0", "x1"], y=["y0", "y1"], line_width=1)

    x_range = y_range = (-0.1875, 1.1875)
    cvs = ds.Canvas(plot_width=11, plot_height=11, x_range=x_range, y_range=y_range)

    # Precalculate expected solutions
    sol_count = nansum(line_antialias_sol_0, line_antialias_sol_1)
    sol_count_intersect = nansum(line_antialias_sol_0_intersect, line_antialias_sol_1)
    sol_min = 3*nanmin(line_antialias_sol_0, line_antialias_sol_1)
    sol_first = 3*np.where(np.isnan(line_antialias_sol_0), line_antialias_sol_1, line_antialias_sol_0)
    sol_last = 3*np.where(np.isnan(line_antialias_sol_1), line_antialias_sol_0, line_antialias_sol_1)

    # Summary of count and sum using self_intersect=True
    agg = cvs.line(
        agg=ds.summary(
            count=ds.count("value", self_intersect=True),
            sum=ds.sum("value", self_intersect=True),
        ), **kwargs)
    assert_eq_ndarray(agg["count"].data, sol_count_intersect, close=True)
    assert_eq_ndarray(agg["sum"].data, 3*sol_count_intersect, close=True)

    # Summary of count and sum using self_intersect=False
    agg = cvs.line(
        agg=ds.summary(
            count=ds.count("value", self_intersect=False),
            sum=ds.sum("value", self_intersect=False),
        ), **kwargs)
    assert_eq_ndarray(agg["count"].data, sol_count, close=True)
    assert_eq_ndarray(agg["sum"].data, 3*sol_count, close=True)

    # Summary of count/sum with mix of self_intersect will force self_intersect=False for both
    agg = cvs.line(
        agg=ds.summary(
            count=ds.count("value", self_intersect=True),
            sum=ds.sum("value", self_intersect=False),
        ), **kwargs)
    assert_eq_ndarray(agg["count"].data, sol_count, close=True)
    assert_eq_ndarray(agg["sum"].data, 3*sol_count, close=True)

    # min, first and last also force use of self_intersect=False
    agg = cvs.line(
        agg=ds.summary(
            count=ds.count("value", self_intersect=True),
            min=ds.min("value"),
        ), **kwargs)
    assert_eq_ndarray(agg["count"].data, sol_count, close=True)
    assert_eq_ndarray(agg["min"].data, sol_min, close=True)

    agg = cvs.line(
        agg=ds.summary(
            count=ds.count("value", self_intersect=True),
            first=ds.first("value"),
        ), **kwargs)
    assert_eq_ndarray(agg["count"].data, sol_count, close=True)
    assert_eq_ndarray(agg["first"].data, sol_first, close=True)

    agg = cvs.line(
        agg=ds.summary(
            count=ds.count("value", self_intersect=True),
            last=ds.last("value"),
        ), **kwargs)
    assert_eq_ndarray(agg["count"].data, sol_count, close=True)
    assert_eq_ndarray(agg["last"].data, sol_last, close=True)

    assert_eq_ndarray(agg.x_range, x_range, close=True)
    assert_eq_ndarray(agg.y_range, y_range, close=True)


line_antialias_nan_sol_intersect = np.array([
    [0.085786, 0.5,      0.085786, nan,      nan,      nan,      nan,      nan,      nan,      0.085786, 0.5,      0.085786],
    [0.5,      1.0,      0.792893, 0.085786, nan,      nan,      nan,      nan,      0.085786, 0.792893, 1.0,      0.5     ],
    [0.085786, 0.792893, 1.0,      0.5,      nan,      nan,      nan,      0.085786, 0.792893, 1.0,      0.792893, 0.085786],
    [nan,      0.085786, 0.5,      0.085786, nan,      nan,      0.085786, 0.792893, 1.0,      0.792893, 0.085786, nan     ],
    [nan,      nan,      nan,      nan,      0.085786, 0.585786, 0.878679, 1.0,      0.792893, 0.085786, nan,      nan     ],
    [nan,      nan,      nan,      nan,      0.585786, 1.792893, 1.792893, 0.878679, 0.085786, nan,      nan,      nan     ],
    [nan,      nan,      nan,      nan,      0.585786, 1.792893, 1.792893, 0.878679, 0.085786, nan,      nan,      nan     ],
    [nan,      nan,      nan,      nan,      0.085786, 0.585786, 0.878679, 1.0,      0.792893, 0.085786, nan,      nan     ],
    [nan,      0.085786, 0.5,      0.085786, nan,      nan,      0.085786, 0.792893, 1.0,      0.792893, 0.085786, nan     ],
    [0.085786, 0.792893, 1.0,      0.5,      nan,      nan,      nan,      0.085786, 0.792893, 1.0,      0.792893, 0.085786],
    [0.5,      1.0,      0.792893, 0.085786, nan,      nan,      nan,      nan,      0.085786, 0.792893, 1.0,      0.5     ],
    [0.085786, 0.5,      0.085786, nan,      nan,      nan,      nan,      nan,      nan,      0.085786, 0.5,      0.085786],
])

line_antialias_nan_sol_max = np.array([
    [0.085786, 0.5,      0.085786, nan,      nan,      nan, nan,      nan,      nan,      0.085786, 0.5,      0.085786],
    [0.5,      1.0,      0.792893, 0.085786, nan,      nan, nan,      nan,      0.085786, 0.792893, 1.0,      0.5     ],
    [0.085786, 0.792893, 1.0,      0.5,      nan,      nan, nan,      0.085786, 0.792893, 1.0,      0.792893, 0.085786],
    [nan,      0.085786, 0.5,      0.085786, nan,      nan, 0.085786, 0.792893, 1.0,      0.792893, 0.085786, nan     ],
    [nan,      nan,      nan,      nan,      0.085786, 0.5, 0.792893, 1.0,      0.792893, 0.085786, nan,      nan     ],
    [nan,      nan,      nan,      nan,      0.5,      1.0, 1.0,      0.792893, 0.085786, nan,      nan,      nan     ],
    [nan,      nan,      nan,      nan,      0.5,      1.0, 1.0,      0.792893, 0.085786, nan,      nan,      nan     ],
    [nan,      nan,      nan,      nan,      0.085786, 0.5, 0.792893, 1.0,      0.792893, 0.085786, nan,      nan     ],
    [nan,      0.085786, 0.5,      0.085786, nan,      nan, 0.085786, 0.792893, 1.0,      0.792893, 0.085786, nan     ],
    [0.085786, 0.792893, 1.0,      0.5,      nan,      nan, nan,      0.085786, 0.792893, 1.0,      0.792893, 0.085786],
    [0.5,      1.0,      0.792893, 0.085786, nan,      nan, nan,      nan,      0.085786, 0.792893, 1.0,      0.5     ],
    [0.085786, 0.5,      0.085786, nan,      nan,      nan, nan,      nan,      nan,      0.085786, 0.5,      0.085786],
])

line_antialias_nan_params = [
    # LineAxis0
    (dict(data=dict(
        x=[0.5, 1.5, np.nan, 4.5, 9.5, np.nan, 0.5, 1.5, np.nan, 4.5, 9.5],
        y=[0.5, 1.5, np.nan, 4.5, 9.5, np.nan, 9.5, 8.5, np.nan, 5.5, 0.5],
    ), dtype='float32'),
    dict(x='x', y='y', axis=0),
    line_antialias_nan_sol_max, line_antialias_nan_sol_intersect),
    # LineAxis0Multi
    (dict(data=dict(
        x0=[0.5, 1.5, np.nan, 4.5, 9.5],
        x1=[0.5, 1.5, np.nan, 4.5, 9.5],
        y0=[0.5, 1.5, np.nan, 4.5, 9.5],
        y1=[9.5, 8.5, np.nan, 5.5, 0.5],
    ), dtype='float32'),
    dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0),
    line_antialias_nan_sol_intersect, line_antialias_nan_sol_intersect),
    # LinesAxis1
    (dict(data=dict(
        x0=[0.5, 0.5],
        x1=[1.5, 1.5],
        x2=[np.nan, np.nan],
        x3=[4.5, 4.5],
        x4=[9.5, 9.5],
        y0=[0.5, 9.5],
        y1=[1.5, 8.5],
        y2=[np.nan, np.nan],
        y3=[4.5, 5.5],
        y4=[9.5, 0.5],
    ), dtype='float32'),
    dict(x=['x0', 'x1', 'x2', 'x3', 'x4'], y=['y0', 'y1', 'y2', 'y3', 'y4'], axis=1),
    line_antialias_nan_sol_intersect, line_antialias_nan_sol_intersect),
    # LinesAxis1XConstant
    (dict(data=dict(
        y0=[0.5, 9.5],
        y1=[1.5, 8.5],
        y2=[np.nan, np.nan],
        y3=[4.5, 5.5],
        y4=[9.5, 0.5],
    ), dtype='float32'),
    dict(x=np.array([0.5, 1.5, np.nan, 4.5, 9.5]), y=['y0', 'y1', 'y2', 'y3', 'y4'], axis=1),
    line_antialias_nan_sol_intersect, line_antialias_nan_sol_intersect),
   # LinesAxis1YConstant
    (dict(data=dict(
        x0=[0.5, 9.5],
        x1=[1.5, 8.5],
        x2=[np.nan, np.nan],
        x3=[4.5, 5.5],
        x4=[9.5, 0.5],
    ), dtype='float32'),
    dict(y=np.array([0.5, 1.5, np.nan, 4.5, 9.5]), x=['x0', 'x1', 'x2', 'x3', 'x4'], axis=1),
    line_antialias_nan_sol_intersect.T, line_antialias_nan_sol_intersect.T),
    # LineAxis1Ragged
    (dict(data=dict(
        x=pd.array([[0.5, 1.5, np.nan, 4.5, 9.5], [0.5, 1.5, np.nan, 4.5, 9.5]], dtype='Ragged[float32]'),
        y=pd.array([[0.5, 1.5, np.nan, 4.5, 9.5], [9.5, 8.5, np.nan, 5.5, 0.5]], dtype='Ragged[float32]'),
    )),
    dict(x='x', y='y', axis=1),
    line_antialias_nan_sol_intersect, line_antialias_nan_sol_intersect),
]
if sp:
    line_antialias_nan_params.append(
        # LineAxis1Geometry
        (
            dict(geom=pd.array(
                [
                    [0.5, 0.5, 1.5, 1.5, np.nan, np.nan, 4.5, 4.5, 9.5, 9.5],
                    [0.5, 9.5, 1.5, 8.5, np.nan, np.nan, 4.5, 5.5, 9.5, 0.5],
                ], dtype='Line[float32]')),
            dict(geometry='geom'),
            line_antialias_nan_sol_intersect, line_antialias_nan_sol_intersect,
        )
    )

@pytest.mark.parametrize('df_kwargs, cvs_kwargs, sol_False, sol_True', line_antialias_nan_params)
@pytest.mark.parametrize('self_intersect', [False, True])
def test_line_antialias_nan(df_kwargs, cvs_kwargs, sol_False, sol_True, self_intersect):
    # Canvas.line() with line_width > 0 has specific identification of start
    # and end line segments from nan coordinates to draw end caps correctly.
    x_range = y_range = (-1, 11)
    cvs = ds.Canvas(plot_width=12, plot_height=12, x_range=x_range, y_range=y_range)

    if 'geometry' in cvs_kwargs:
        df = sp.GeoDataFrame(df_kwargs)
    else:
        df = pd.DataFrame(**df_kwargs)

    agg = cvs.line(df, line_width=2, agg=ds.count(self_intersect=self_intersect), **cvs_kwargs)
    sol = sol_True if self_intersect else sol_False
    assert_eq_ndarray(agg.data, sol, close=True)
    assert_eq_ndarray(agg.x_range, x_range, close=True)
    assert_eq_ndarray(agg.y_range, y_range, close=True)


def test_line_antialias_categorical():
    df = pd.DataFrame(dict(
        x=np.asarray([0, 1, 1, 0, np.nan, 0, 1/3.0, 2/3.0, 1]),
        y=np.asarray([0, 1, 0, 1, np.nan, 0.125, 0.15, 0.175, 0.2]),
        cat=[1, 1, 1, 1, 1, 2, 2, 2, 2],
    ))
    df["cat"] = df["cat"].astype("category")

    x_range = y_range = (-0.1875, 1.1875)
    cvs = ds.Canvas(plot_width=11, plot_height=11, x_range=x_range, y_range=y_range)

    agg = cvs.line(source=df, x="x", y="y", line_width=1,
                   agg=ds.by("cat", ds.count(self_intersect=False)))
    assert_eq_ndarray(agg.data[:, :, 0], line_antialias_sol_0, close=True)
    assert_eq_ndarray(agg.data[:, :, 1], line_antialias_sol_1, close=True)

    agg = cvs.line(source=df, x="x", y="y", line_width=1,
                   agg=ds.by("cat", ds.count(self_intersect=True)))
    assert_eq_ndarray(agg.data[:, :, 0], line_antialias_sol_0_intersect, close=True)
    assert_eq_ndarray(agg.data[:, :, 1], line_antialias_sol_1, close=True)


@pytest.mark.parametrize('self_intersect', [False, True])
def test_line_antialias_duplicate_points(self_intersect):
    # Issue #1098. Duplicate points should not raise a divide by zero error and
    # should produce same results as without duplicate points.
    cvs = ds.Canvas(plot_width=10, plot_height=10, x_range=(-0.1, 1.1), y_range=(0.9, 2.1))

    df = pd.DataFrame(dict(x=[0, 1], y=[1, 2]))
    agg_no_duplicate = cvs.line(source=df, x="x", y="y", line_width=1,
                                agg=ds.count(self_intersect=self_intersect))

    df = pd.DataFrame(dict(x=[0, 0, 1], y=[1, 1, 2]))
    agg_duplicate = cvs.line(source=df, x="x", y="y", line_width=1,
                             agg=ds.count(self_intersect=self_intersect))

    assert_eq_xr(agg_no_duplicate, agg_duplicate)


@pytest.mark.parametrize('reduction', [
    ds.std('value'),
    ds.var('value'),
])
def test_line_antialias_reduction_not_implemented(reduction):
    # Issue #1133, detect and report reductions that are not implemented.
    cvs = ds.Canvas(plot_width=10, plot_height=10)
    df = pd.DataFrame(dict(x=[0, 1], y=[1, 2], value=[1, 2]))

    with pytest.raises(NotImplementedError):
        cvs.line(df, 'x', 'y', line_width=1, agg=reduction)


def test_line_antialias_where():
    df = pd.DataFrame(dict(
        y0 = [0.5, 1.0, 0.0],
        y1 = [1.0, 0.0, 0.5],
        y2 = [0.0, 0.5, 1.0],
        value = [2.2, 3.3, 1.1],
        other = [-9.0, -7.0, -5.0],
    ))
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


@pytest.mark.parametrize('reduction,dtype,aa_dtype', [
    (ds.any(), bool, np.float32),
    (ds.count(), np.uint32, np.float32),
    (ds.max("value"), np.float64, np.float64),
    (ds.min("value"), np.float64, np.float64),
    (ds.sum("value"), np.float64, np.float64),
    (ds.where(ds.max("value")), np.int64, np.int64),
    (ds.where(ds.max("value"), "other"), np.float64, np.float64),
])
def test_reduction_dtype(reduction, dtype, aa_dtype):
    cvs = ds.Canvas(plot_width=10, plot_height=10)
    df = pd.DataFrame(dict(x=[0, 1], y=[1, 2], value=[1, 2], other=[1.2, 3.4]))

    # Non-antialiased lines
    agg = cvs.line(df, 'x', 'y', line_width=0, agg=reduction)
    assert agg.dtype == dtype

    # Antialiased lines
    if not isinstance(reduction, ds.where):  # Antialiased ds.where not implemented")
        agg = cvs.line(df, 'x', 'y', line_width=1, agg=reduction)
        assert agg.dtype == aa_dtype


@pytest.mark.parametrize('df', dfs)
@pytest.mark.parametrize('canvas', [
    ds.Canvas(x_axis_type='log'),
    ds.Canvas(x_axis_type='log', x_range=(0, 1)),
    ds.Canvas(y_axis_type='log'),
    ds.Canvas(y_axis_type='log', y_range=(0, 1)),
])
def test_log_axis_not_positive(df, canvas):
    with pytest.raises(ValueError, match='Range values must be >0 for logarithmic axes'):
        canvas.line(df, 'x', 'y')


@pytest.mark.parametrize('selector', [
    ds.any(),
    ds.count(),
    ds.mean('value'),
    ds.std('value'),
    ds.sum('value'),
    ds.summary(any=ds.any()),
    ds.var('value'),
    ds.where(ds.max('value'), 'other'),
])
def test_where_unsupported_selector(selector):
    cvs = ds.Canvas(plot_width=10, plot_height=10)
    df = pd.DataFrame(dict(x=[0, 1], y=[1, 2], value=[1, 2], ))

    with pytest.raises(TypeError, match='selector can only be a first, first_n, last, last_n, '
                                        'max, max_n, min or min_n reduction'):
        cvs.line(df, 'x', 'y', agg=ds.where(selector, 'value'))


def test_line_coordinate_lengths():
    # Issue #1159.
    cvs = ds.Canvas(plot_width=10, plot_height=6)
    msg = r'^x and y coordinate lengths do not match'

    # LineAxis0Multi (axis=0) and LinesAxis1 (axis=1)
    df = pd.DataFrame(
        dict(x0=[0, 0.2, 1], y0=[0, 0.4, 1], x1=[0, 0.6, 1], y1=[1, 0.8, 1]))
    for axis in (0, 1):
        with pytest.raises(ValueError, match=msg):
            cvs.line(source=df, x=["x0"], y=["y0", "y1"], axis=axis)
        with pytest.raises(ValueError, match=msg):
            cvs.line(source=df, x=["x0", "x1"], y=["y0"], axis=axis)

    # LinesAxis1XConstant
    df = pd.DataFrame(dict(y0=[0, 1, 0, 1], y1=[0, 1, 1, 0]))
    for nx in (1, 3):
        with pytest.raises(ValueError, match=msg):
            cvs.line(source=df, x=np.arange(nx), y=["y0", "y1"], axis=1)

    # LinesAxis1YConstant
    df = pd.DataFrame(dict(x0=[0, 1, 0, 1], x1=[0, 1, 1, 0]))
    for ny in (1, 3):
        with pytest.raises(ValueError, match=msg):
            cvs.line(source=df, x=["x0", "x1"], y=np.arange(ny), axis=1)


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

    for cvs in cvs_list:
        with pytest.raises(ValueError, match=msg):
            cvs.points(df, "x", "y", ds.mean("z"))


@pytest.mark.parametrize('df', dfs)
def test_categorical_where_max(df):
    sol_rowindex = xr.DataArray([[[4, 1, -1, 3], [12, 13, 14, 11]], [[8, 5, 6, 7], [16, 17, 18, 15]]],
                                coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)

    # Using row index
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.max('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)

    # Using another column
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.max('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)


@pytest.mark.parametrize('df', dfs)
def test_categorical_where_min(df):
    sol_rowindex = xr.DataArray([[[0, 1, -1, 3], [12, 13, 10, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]],
                                coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)

    # Using row index
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.min('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)

    # Using another column
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.min('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)


@pytest.mark.parametrize('df', dfs)
def test_categorical_where_first(df):
    sol_rowindex = xr.DataArray([[[0, 1, -1, 3], [12, 13, 10, 11]], [[8, 5, 6, 7], [16, 17, 18, 15]]],
                                coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)

    # Using row index
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.first('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)

    # Using another column
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.first('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)


@pytest.mark.parametrize('df', dfs)
def test_categorical_where_last(df):
    sol_rowindex = xr.DataArray([[[4, 1, -1, 3], [12, 13, 14, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]],
                                coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)

    # Using row index
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)

    # Using another column
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)


@pytest.mark.parametrize('df', dfs)
def test_categorical_where_max_n(df):
    sol_rowindex = xr.DataArray(
        [[[[4, 0, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]],
          [[12, -1, -1], [13, -1, -1], [14, 10, -1], [11, -1, -1]]],
         [[[8, -1, -1], [5, 9, -1], [6, -1, -1], [7, -1, -1]],
          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [15, 19, -1]]]],
        coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)

    for n in range(1, 4):
        # Using row index
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.max_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.max('plusminus')))).data)

        # Using another column
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.max_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.max('plusminus'), 'reverse'))).data)


@pytest.mark.parametrize('df', dfs)
def test_categorical_where_min_n(df):
    sol_rowindex = xr.DataArray(
        [[[[0, 4, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]],
          [[12, -1, -1], [13, -1, -1], [10, 14, -1], [11, -1, -1]]],
         [[[8, -1, -1], [9, 5, -1], [6, -1, -1], [7, -1, -1]],
          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [19, 15, -1]]]],
        coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)

    for n in range(1, 4):
        # Using row index
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.min_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.min('plusminus')))).data)

        # Using another column
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.min_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.min('plusminus'), 'reverse'))).data)


@pytest.mark.parametrize('df', dfs)
def test_categorical_where_first_n(df):
    sol_rowindex = xr.DataArray(
        [[[[0, 4, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]],
          [[12, -1, -1], [13, -1, -1], [10, 14, -1], [11, -1, -1]]],
         [[[8, -1, -1], [5, 9, -1], [6, -1, -1], [7, -1, -1]],
          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [15, 19, -1]]]],
        coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)

    for n in range(1, 4):
        # Using row index
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.first_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.first('plusminus')))).data)

        # Using another column
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.first_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.first('plusminus'), 'reverse'))).data)


@pytest.mark.parametrize('df', dfs)
def test_categorical_where_last_n(df):
    sol_rowindex = xr.DataArray(
        [[[[4, 0, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]],
          [[12, -1, -1], [13, -1, -1], [14, 10, -1], [11, -1, -1]]],
         [[[8, -1, -1], [9, 5, -1], [6, -1, -1], [7, -1, -1]],
          [[16, -1, -1], [17, -1, -1], [18, -1, -1], [19, 15, -1]]]],
        coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)

    for n in range(1, 4):
        # Using row index
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last('plusminus')))).data)

        # Using another column
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data,
                              c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last('plusminus'), 'reverse'))).data)
