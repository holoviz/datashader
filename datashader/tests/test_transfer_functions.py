from __future__ import annotations

from io import BytesIO

import numpy as np
import xarray as xr
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close

try:
    import dask.array as da
except ImportError:
    da = None

coords = dict([('x_axis', [3, 4, 5]), ('y_axis', [0, 1, 2])])
dims = ['y_axis', 'x_axis']

# CPU
def build_agg(array_module=np):
    a = array_module.arange(10, 19, dtype='u4').reshape((3, 3))
    a[[0, 1, 2], [0, 1, 2]] = 0
    s_a = xr.DataArray(a, coords=coords, dims=dims)
    b = array_module.arange(10, 19, dtype='f4').reshape((3, 3))
    b[[0, 1, 2], [0, 1, 2]] = array_module.nan
    s_b = xr.DataArray(b, coords=coords, dims=dims)
    c = array_module.arange(10, 19, dtype='f8').reshape((3, 3))
    c[[0, 1, 2], [0, 1, 2]] = array_module.nan
    s_c = xr.DataArray(c, coords=coords, dims=dims)
    d = array_module.arange(10, 19, dtype='u4').reshape((3, 3))
    d[[0, 1, 2, 2], [0, 1, 2, 1]] = 1
    s_d = xr.DataArray(d, coords=coords, dims=dims)
    agg = xr.Dataset(dict(a=s_a, b=s_b, c=s_c, d=s_d))
    return agg


def build_agg_dask():
    # Dask arrays are immutable `build_agg(da)` won't work.
    # Create numpy based DataArray and convert to Dask by forcing chunking.
    if da is None:
        pytest.skip("dask is not available")
    return build_agg(np).chunk({d: 1 for d in dims})


def build_agg_cupy():
    import cupy
    return build_agg(cupy)


_backends = [
    pytest.param(build_agg, id="numpy"),
    pytest.param(build_agg_dask, id="dask"),
    pytest.param(build_agg_cupy, marks=pytest.mark.gpu, id="cupy"),
]

@pytest.fixture(params=_backends)
def agg(request):
    return request.param()


def create_dask_array(*args, **kwargs):
    """Create a dask array wrapping around a numpy array."""
    if da is None:
        pytest.skip("dask is not available")
    return da.from_array(np.array(*args, **kwargs))


def create_cupy_array(*args, **kwargs):
    import cupy
    return cupy.array(*args, **kwargs)


_backends = [
    pytest.param(np.array, id="numpy"),
    pytest.param(create_dask_array, id="dask"),
    pytest.param(create_cupy_array, marks=pytest.mark.gpu, id="cupy"),
]

@pytest.fixture(params=_backends)
def array(request):
    return request.param

int_span = [11, 17]
float_span = [11.0, 17.0]

solution_lists = {
    'log':
        [[0, 4291543295, 4286741503],
         [4283978751, 0, 4280492543],
         [4279242751, 4278190335, 0]],
    'cbrt':
        [[0, 4291543295, 4284176127],
         [4282268415, 0, 4279834879],
         [4278914047, 4278190335, 0]],
    'linear':
        [[0, 4291543295, 4289306879],
         [4287070463, 0, 4282597631],
         [4280361215, 4278190335, 0]]
}

solutions = {how: tf.Image(np.array(v, dtype='u4'),
                           coords=coords, dims=dims)
             for how, v in solution_lists.items()}

# Same result obtained regardless of data dtype (u4, f4, f8)
eq_hist_sol = np.array([[0, 4291543295, 4288846335],
                        [4286149631, 0, 4283518207],
                        [4280821503, 4278190335, 0]], dtype='u4')

eq_hist_sol_rescale_discrete_levels = {
    'a': np.array([[0, 4289306879, 4287070463],
                   [4284834047, 0, 4282597631],
                   [4280361215, 4278190335, 0]], dtype='u4'),
    'b': np.array([[0, 4289306879, 4287070207],
                   [4284834047, 0, 4282597375],
                   [4280361215, 4278190335, 0]], dtype='u4')}
eq_hist_sol_rescale_discrete_levels['c'] = eq_hist_sol_rescale_discrete_levels['b']


def check_span(x, cmap, how, sol):
    # Copy inputs that will be modified
    sol = sol.copy()
    if isinstance(x, xr.DataArray) and da and isinstance(x.data, da.Array):
        x = x.compute()
    else:
        x = x.copy()

    # All data no span
    img = tf.shade(x, cmap=cmap, how=how, span=None)
    assert_eq_xr(img, sol)

    # All data with span
    img = tf.shade(x, cmap=cmap, how=how, span=float_span)
    assert_eq_xr(img, sol)

    # Decrease smallest. This value should be clipped to span[0] and the
    # resulting image should be identical
    x[0, 1] = 10
    x_input = x.copy()
    img = tf.shade(x, cmap=cmap, how=how, span=float_span)
    assert_eq_xr(img, sol)

    # Check that clipping doesn't alter input array
    x.equals(x_input)

    # Increase largest. This value should be clipped to span[1] and the
    # resulting image should be identical
    x[2, 1] = 18
    x_input = x.copy()
    img = tf.shade(x, cmap=cmap, how=how, span=float_span)
    assert_eq_xr(img, sol)

    # Check that clipping doesn't alter input array
    x.equals(x_input)

    # zero out smallest. If span is working properly the zeroed out pixel
    # will be masked out and all other pixels will remain unchanged
    x[0, 1] = 0 if x.dtype.kind in ('i', 'u') else np.nan
    img = tf.shade(x, cmap=cmap, how=how, span=float_span)
    sol[0, 1] = sol[0, 0]
    assert_eq_xr(img, sol)

    # zero out the largest value
    x[2, 1] = 0 if x.dtype.kind in ('i', 'u') else np.nan
    img = tf.shade(x, cmap=cmap, how=how, span=float_span)
    sol[2, 1] = sol[0, 0]
    assert_eq_xr(img, sol)


@pytest.mark.parametrize('attr', ['a', 'b', 'c'])
@pytest.mark.parametrize('span', [None, int_span, float_span])
def test_shade(agg, attr, span):
    x = getattr(agg, attr)
    cmap = ['pink', 'red']

    img = tf.shade(x, cmap=cmap, how='log', span=span)
    sol = solutions['log']
    assert_eq_xr(img, sol)
    # Check dims/coordinates order
    assert list(img.coords) == ['x_axis', 'y_axis']
    assert list(img.dims) == ['y_axis', 'x_axis']

    img = tf.shade(x, cmap=cmap, how='cbrt', span=span)
    sol = solutions['cbrt']
    assert_eq_xr(img, sol)

    img = tf.shade(x, cmap=cmap, how='linear', span=span)
    sol = solutions['linear']
    assert_eq_xr(img, sol)

    # span option not supported with how='eq_hist'
    if span is None:
        img = tf.shade(x, cmap=cmap, how='eq_hist', rescale_discrete_levels=False)
        sol = tf.Image(eq_hist_sol, coords=coords, dims=dims)
        assert_eq_xr(img, sol)

        img = tf.shade(x, cmap=cmap, how='eq_hist', rescale_discrete_levels=True)
        sol = tf.Image(eq_hist_sol_rescale_discrete_levels[attr], coords=coords, dims=dims)
        assert_image_close(img, sol, tolerance=1)

    img = tf.shade(x, cmap=cmap,
                   how=lambda x, mask: np.where(mask, np.nan, x ** 2))
    sol = np.array([[0, 4291543295, 4291148543],
                    [4290030335, 0, 4285557503],
                    [4282268415, 4278190335, 0]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)


@pytest.mark.parametrize('attr', ['a', 'b', 'c'])
@pytest.mark.parametrize('how', ['linear', 'log', 'cbrt'])
@pytest.mark.parametrize('cmap', [['pink', 'red'], ('#FFC0CB', '#FF0000')])
def test_span_cmap_list(agg, attr, how, cmap):
    # Get input
    x = getattr(agg, attr).copy()

    # Get expected solution for interpolation method
    sol = solutions[how]

    # Check span
    check_span(x, cmap, how, sol)


@pytest.mark.parametrize('cmap', ['black', (0, 0, 0), '#000000'])
def test_span_cmap_single(agg, cmap):
    # Get input
    x = agg.a

    # Build expected solution DataArray
    sol = np.array([[0, 671088640, 1946157056],
                    [2701131776, 0, 3640655872],
                    [3976200192, 4278190080, 0]])
    sol = tf.Image(sol, coords=coords, dims=dims)

    # Check span
    check_span(x, cmap, 'log', sol)


def test_span_cmap_mpl(agg):
    # Get inputs
    x = agg.a

    # Get MPL colormap
    cm = pytest.importorskip('matplotlib.cm')
    cmap = cm.viridis

    # Build expected solution Data Array
    sol = np.array([[0, 4283695428, 4287524142],
                    [4287143710, 0, 4282832267],
                    [4280213706, 4280608765, 0]])
    sol = tf.Image(sol, coords=coords, dims=dims)

    # Check span
    check_span(x, cmap, 'log', sol)


def test_shade_bool():
    data = ~np.eye(3, dtype='bool')
    x = tf.Image(data, coords=coords, dims=dims)
    sol = tf.Image(np.where(data, 4278190335, 0).astype('uint32'),
                       coords=coords, dims=dims)
    img = tf.shade(x, cmap=['pink', 'red'], how='log')
    assert_eq_xr(img, sol)
    img = tf.shade(x, cmap=['pink', 'red'], how='cbrt')
    assert_eq_xr(img, sol)
    img = tf.shade(x, cmap=['pink', 'red'], how='linear')
    assert_eq_xr(img, sol)
    img = tf.shade(x, cmap=['pink', 'red'], how='eq_hist')
    assert_eq_xr(img, sol)


def test_shade_cmap(agg):
    cmap = ['red', (0, 255, 0), '#0000FF']
    img = tf.shade(agg.a, how='log', cmap=cmap)
    sol = np.array([[0, 4278190335, 4278236489],
                    [4280344064, 0, 4289091584],
                    [4292225024, 4294901760, 0]])
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)


@pytest.mark.parametrize('cmap', ['black', (0, 0, 0), '#000000'])
def test_shade_cmap_non_categorical_alpha(agg, cmap):
    img = tf.shade(agg.a, how='log', cmap=cmap)
    sol = np.array([[         0,  671088640, 1946157056],
                    [2701131776,          0, 3640655872],
                    [3976200192, 4278190080,          0]])
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)


def test_shade_cmap_errors(agg):
    with pytest.raises(ValueError):
        tf.shade(agg.a, cmap='foo')

    with pytest.raises(ValueError):
        tf.shade(agg.a, cmap=[])


def test_shade_mpl_cmap(agg):
    cm = pytest.importorskip('matplotlib.cm')
    img = tf.shade(agg.a, how='log', cmap=cm.viridis)
    sol = np.array([[0, 4283695428, 4287524142],
                    [4287143710, 0, 4282832267],
                    [4280213706, 4280608765, 0]])
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)


def test_shade_category(array):
    coords = [np.array([0, 1]), np.array([2, 5])]
    cat_agg = tf.Image(array([[(0, 12, 0), (3, 0, 3)], [(12, 12, 12), (24, 0, 0)]], dtype='u4'),
                       coords=(coords + [['a', 'b', 'c']]),
                       dims=(dims + ['cats']))

    colors = [(255, 0, 0), '#0000FF', 'orange']

    img = tf.shade(cat_agg, color_key=colors, how='log', min_alpha=20)
    sol = np.array([[2583625728, 335565567],
                    [4283774890, 3707764991]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    # Check dims/coordinates order
    assert list(img.coords) == ['x_axis', 'y_axis']
    assert list(img.dims) == ['y_axis', 'x_axis']

    colors = dict(zip('abc', colors))

    img = tf.shade(cat_agg, color_key=colors, how='cbrt', min_alpha=20)
    sol = np.array([[2650734592, 335565567],
                    [4283774890, 3657433343]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)

    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20)
    sol = np.array([[1140785152, 335565567],
                    [4283774890, 2701132031]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)

    img = tf.shade(cat_agg, color_key=colors,
                   how=lambda x, m: np.where(m, np.nan, x) ** 2,
                   min_alpha=20)
    sol = np.array([[503250944, 335565567],
                    [4283774890, 1744830719]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)

    # all pixels should be at min_alpha
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=0, span=(50, 100))
    sol = np.array([[16711680, 21247],
                    [5584810, 255]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    # redundant verification that alpha channel is all 0x00
    assert ((img.data[0,0] >> 24) & 0xFF) == 0
    assert ((img.data[0,1] >> 24) & 0xFF) == 0
    assert ((img.data[1,0] >> 24) & 0xFF) == 0
    assert ((img.data[1,1] >> 24) & 0xFF) == 0

    # all pixels should be at max_alpha
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=0, span=(0, 2))
    sol = np.array([[4294901760, 4278211327],
                    [4283774890, 4278190335]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    # redundant verification that alpha channel is all 0xFF
    assert ((img.data[0,0] >> 24) & 0xFF) == 255
    assert ((img.data[0,1] >> 24) & 0xFF) == 255
    assert ((img.data[1,0] >> 24) & 0xFF) == 255
    assert ((img.data[1,1] >> 24) & 0xFF) == 255

    # One pixel should be min-alpha, the other max-alpha
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=0, span=(6, 36))
    sol = np.array([[872349696, 21247],
                    [4283774890, 2566914303]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    # redundant verification that alpha channel is correct
    assert ((img.data[0,0] >> 24) & 0xFF) == 51 # (6 / 30) * 255
    assert ((img.data[0,1] >> 24) & 0xFF) == 0
    assert ((img.data[1,0] >> 24) & 0xFF) == 255
    assert ((img.data[1,1] >> 24) & 0xFF) == 153 # ( 18 /30) * 255

    # One pixel should be min-alpha, the other max-alpha
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=0, span=(0, 72))
    sol = np.array([[721354752, 352342783],
                    [2136291242, 1426063615]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    # redundant verification that alpha channel is correct
    assert ((img.data[0,0] >> 24) & 0xFF) == 42 # (12 / 72) * 255
    assert ((img.data[0,1] >> 24) & 0xFF) == 21 # (6 / 72) * 255
    assert ((img.data[1,0] >> 24) & 0xFF) == 127 # ( 36 / 72) * 255
    assert ((img.data[1,1] >> 24) & 0xFF) == 85 # ( 24 /72 ) * 255

    # test that empty coordinates are always fully transparent, even when
    # min_alpha is non-zero
    cat_agg = tf.Image(array([[(0, 0, 0), (3, 0, 3)],
                              [(12, 12, 12), (24, 0, 0)]], dtype='u4'),
                           coords=(coords + [['a', 'b', 'c']]),
                           dims=(dims + ['cats']))

    # First test auto-span
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20)
    sol = np.array([[5584810, 335565567],
                    [4283774890, 2701132031]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)

    # redundant verification that alpha channel is correct
    assert ((img.data[0,0] >> 24) & 0xFF) == 0 # fully transparent
    assert ((img.data[0,1] >> 24) & 0xFF) != 0 # not fully transparent
    assert ((img.data[1,0] >> 24) & 0xFF) != 0 # not fully transparent
    assert ((img.data[1,1] >> 24) & 0xFF) != 0 # not fully transparent

    # Next test manual-span
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20, span=(6, 36))
    sol = np.array([[5584810, 335565567],
                    [4283774890, 2701132031]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)

    # redundant verification that alpha channel is correct
    assert ((img.data[0,0] >> 24) & 0xFF) == 0 # fully transparent
    assert ((img.data[0,1] >> 24) & 0xFF) != 0 # not fully transparent
    assert ((img.data[1,0] >> 24) & 0xFF) != 0 # not fully transparent
    assert ((img.data[1,1] >> 24) & 0xFF) != 0 # not fully transparent


    # Categorical aggregations with some reductions (such as sum) can result in negative
    # values in the data here we test positive and negative values
    cat_agg = tf.Image(array([[(0, -30, 0), (18, 0, -18)],
                              [(-2, 2, -2), (-18, 9, 12)]], dtype='i4'),
                       coords=(coords + [['a', 'b', 'c']]),
                       dims=(dims + ['cats']))

    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20)
    sol = np.array([[335565567, 3914667690],
                    [3680253090, 4285155988]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert ((img.data[0,0] >> 24) & 0xFF) == 20
    assert ((img.data[0,1] >> 24) & 0xFF) == 233
    assert ((img.data[1,0] >> 24) & 0xFF) == 219
    assert ((img.data[1,1] >> 24) & 0xFF) == 255

    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20, span=(0, 3))
    sol = np.array([[335565567, 341120682],
                    [341587106, 4285155988]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert ((img.data[0,0] >> 24) & 0xFF) == 20 # min alpha
    assert ((img.data[0,1] >> 24) & 0xFF) == 20 # min alpha
    assert ((img.data[1,0] >> 24) & 0xFF) == 20 # min alpha
    assert ((img.data[1,1] >> 24) & 0xFF) == 255

    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20, color_baseline=9)
    sol = np.array([[341129130, 3909091583],
                    [3679795114, 4278232575]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert ((img.data[0,0] >> 24) & 0xFF) == 20
    assert ((img.data[0,1] >> 24) & 0xFF) == 233
    assert ((img.data[1,0] >> 24) & 0xFF) == 219
    assert ((img.data[1,1] >> 24) & 0xFF) == 255

    # Categorical aggregations with some reductions (such as sum) can result in negative
    # values in the data, here we test all negative values
    cat_agg = tf.Image(array([[(0, -30, 0), (-18, 0, -18)],
                              [(-2, -2, -2), (-18, 0, 0)]], dtype='i4'),
                       coords=(coords + [['a', 'b', 'c']]),
                       dims=(dims + ['cats']))

    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20)
    sol = np.array([[1124094719, 344794225],
                    [4283774890, 2708096148]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert ((img.data[0,0] >> 24) & 0xFF) == 67
    assert ((img.data[0,1] >> 24) & 0xFF) == 20
    assert ((img.data[1,0] >> 24) & 0xFF) == 255
    assert ((img.data[1,1] >> 24) & 0xFF) == 161

    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20, span=(6, 36))
    sol = np.array([[335565567, 344794225],
                    [341129130, 342508692]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert ((img.data[0,0] >> 24) & 0xFF) == 20 # min alpha
    assert ((img.data[0,1] >> 24) & 0xFF) == 20 # min alpha
    assert ((img.data[1,0] >> 24) & 0xFF) == 20 # min alpha
    assert ((img.data[1,1] >> 24) & 0xFF) == 20 # min alpha


def test_shade_zeros(array):
    coords = [np.array([0, 1]), np.array([2, 5])]
    cat_agg = tf.Image(array([[(0, 0, 0), (0, 0, 0)],
                              [(0, 0, 0), (0, 0, 0)]], dtype='u4'),
                           coords=(coords + [['a', 'b', 'c']]),
                           dims=(dims + ['cats']))

    colors = [(255, 0, 0), '#0000FF', 'orange']

    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=0)
    sol = np.array([[5584810, 5584810],
                    [5584810, 5584810]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)


@pytest.mark.parametrize('attr', ['d'])
@pytest.mark.parametrize('rescale', [False, True])
def test_shade_rescale_discrete_levels(agg, attr, rescale):
    x = getattr(agg, attr)
    cmap = ['pink', 'red']
    img = tf.shade(x, cmap=cmap, how='eq_hist', rescale_discrete_levels=rescale)
    if rescale:
        sol = np.array([[0xff8981ff, 0xff6d67ff, 0xff524dff],
                        [0xff3633ff, 0xff8981ff, 0xff1b19ff],
                        [0xff0000ff, 0xff8981ff, 0xff8981ff]], dtype='uint32')
    else:
        sol = np.array([[0xffcbc0ff, 0xffa299ff, 0xff7973ff],
                        [0xff514cff, 0xffcbc0ff, 0xff2826ff],
                        [0xff0000ff, 0xffcbc0ff, 0xffcbc0ff]], dtype='uint32')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)


def test_shade_rescale_discrete_levels_categorical(array):
    arr = array([[[1, 2], [0, 1]],
                              [[0, 0], [0, 0]],
                              [[1, 0], [3, 0]],
                              [[1, 0], [2, 1]]], dtype='u4')
    agg = xr.DataArray(data=arr, coords=dict(y=[0, 1, 2, 3], x=[0, 1], cat=['a', 'b']))
    img = tf.shade(agg, how='eq_hist', rescale_discrete_levels=True)

    sol = np.array([[0xff845c70, 0x6fb87e37],
                    [0x006a4c8d, 0x006a4c8d],
                    [0x6f1c1ae4, 0xff1c1ae4],
                    [0x6f1c1ae4, 0xff503baa]])
    assert_eq_ndarray(img.data, sol)


@pytest.mark.parametrize('empty_array', [
    np.zeros((2, 2, 2), dtype=np.uint32),
    np.full((2, 2, 2), np.nan, dtype=np.float64),
])
@pytest.mark.parametrize('on_gpu', [False, pytest.param(True, marks=pytest.mark.gpu)])
def test_shade_all_masked(empty_array, on_gpu):
    # Issue #1166, return early with array of all nans if all of data is masked out.
    # Before the fix this test results in:
    #   IndexError: index -1 is out of bounds for axis 0 with size 0
    if on_gpu:
        import cupy
        empty_array = cupy.array(empty_array)
    agg = xr.DataArray(
        data=empty_array,
        coords=dict(y=[0, 1], x=[0, 1], cat=['a', 'b']),
    )
    im = tf.shade(agg, how='eq_hist', cmap=["white", "white"])
    assert isinstance(im.data, np.ndarray)
    assert im.shape == (2, 2)


coords2 = [np.array([0, 2]), np.array([3, 5])]
img1 = tf.Image(np.array([[0xff00ffff, 0x00000000],
                          [0x00000000, 0xff00ff7d]], dtype='uint32'),
                coords=coords2, dims=dims)
img2 = tf.Image(np.array([[0x00000000, 0x00000000],
                          [0x000000ff, 0x7d7d7dff]], dtype='uint32'),
                coords=coords2, dims=dims)


def test_set_background():
    out = tf.set_background(img1)
    assert out.equals(img1)
    sol = tf.Image(np.array([[0xff00ffff, 0xff0000ff],
                             [0xff0000ff, 0xff00ff7d]], dtype='uint32'),
                   coords=coords2, dims=dims)
    out = tf.set_background(img1, 'red')
    assert out.equals(sol)


def test_stack():
    img = tf.stack(img1, img2)
    out = np.array([[0xff00ffff, 0x00000000],
                    [0x00000000, 0xff3dbfbc]], dtype='uint32')
    assert (img.x_axis == img1.x_axis).all()
    assert (img.y_axis == img1.y_axis).all()
    np.testing.assert_equal(img.data, out)

    img = tf.stack(img2, img1)
    out = np.array([[0xff00ffff, 0x00000000],
                    [0x00000000, 0xff00ff7d]], dtype='uint32')
    assert (img.x_axis == img1.x_axis).all()
    assert (img.y_axis == img1.y_axis).all()
    np.testing.assert_equal(img.data, out)

    img = tf.stack(img1, img2, how='add')
    out = np.array([[0xff00ffff, 0x00000000],
                    [0x00000000, 0xff3dfffa]], dtype='uint32')
    assert (img.x_axis == img1.x_axis).all()
    assert (img.y_axis == img1.y_axis).all()
    np.testing.assert_equal(img.data, out)


def test_masks():
    # Square
    mask = tf._square_mask(2)
    np.testing.assert_equal(mask, np.ones((5, 5), dtype='bool'))
    np.testing.assert_equal(tf._square_mask(0), np.ones((1, 1), dtype='bool'))
    # Circle
    np.testing.assert_equal(tf._circle_mask(0), np.ones((1, 1), dtype='bool'))
    out = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]], dtype='bool')
    np.testing.assert_equal(tf._circle_mask(1), out)
    out = np.array([[0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0, 0]], dtype='bool')
    np.testing.assert_equal(tf._circle_mask(3), out)


def test_rgb_spread():
    p = 0x7d00007d
    g = 0x7d00FF00
    b = 0x7dFF0000
    data = np.array([[p, p, 0, 0, 0],
                     [p, g, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, b, 0],
                     [0, 0, 0, 0, 0]], dtype='uint32')
    coords = [np.arange(5), np.arange(5)]
    img = tf.Image(data, coords=coords, dims=dims)

    s = tf.spread(img)
    o = np.array([[0xed00863b, 0xed00863b, 0xbc00a82a, 0x00000000, 0x00000000],
                  [0xed00863b, 0xed00863b, 0xbc00a82a, 0x00000000, 0x00000000],
                  [0xbc00a82a, 0xbc00a82a, 0xbca85600, 0x7dff0000, 0x7dff0000],
                  [0x00000000, 0x00000000, 0x7dff0000, 0x7dff0000, 0x7dff0000],
                  [0x00000000, 0x00000000, 0x7dff0000, 0x7dff0000, 0x7dff0000]])
    np.testing.assert_equal(s.data, o)
    assert (s.x_axis == img.x_axis).all()
    assert (s.y_axis == img.y_axis).all()
    assert s.dims == img.dims

    s = tf.spread(img, px=2)
    o = np.array([[0xed00863b, 0xed00863b, 0xed00863b, 0xbc00a82a, 0x00000000],
                  [0xed00863b, 0xed00863b, 0xf581411c, 0xdc904812, 0x7dff0000],
                  [0xed00863b, 0xf581411c, 0xed864419, 0xbca85600, 0x7dff0000],
                  [0xbc00a82a, 0xdc904812, 0xbca85600, 0x7dff0000, 0x7dff0000],
                  [0x00000000, 0x7dff0000, 0x7dff0000, 0x7dff0000, 0x7dff0000]])
    np.testing.assert_equal(s.data, o)

    s = tf.spread(img, shape='square')
    o = np.array([[0xed00863b, 0xed00863b, 0xbc00a82a, 0x00000000, 0x00000000],
                  [0xed00863b, 0xed00863b, 0xbc00a82a, 0x00000000, 0x00000000],
                  [0xbc00a82a, 0xbc00a82a, 0xbca85600, 0x7dff0000, 0x7dff0000],
                  [0x00000000, 0x00000000, 0x7dff0000, 0x7dff0000, 0x7dff0000],
                  [0x00000000, 0x00000000, 0x7dff0000, 0x7dff0000, 0x7dff0000]])
    np.testing.assert_equal(s.data, o)

    s = tf.spread(img, how='add')
    o = np.array([[0xff007db7, 0xff007db7, 0xfa007f3e, 0x00000000, 0x00000000],
                  [0xff007db7, 0xff007db7, 0xfa007f3e, 0x00000000, 0x00000000],
                  [0xfa007f3e, 0xfa007f3e, 0xfa7f7f00, 0x7dff0000, 0x7dff0000],
                  [0x00000000, 0x00000000, 0x7dff0000, 0x7dff0000, 0x7dff0000],
                  [0x00000000, 0x00000000, 0x7dff0000, 0x7dff0000, 0x7dff0000]])
    np.testing.assert_equal(s.data, o)

    mask = np.array([[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]])
    s = tf.spread(img, mask=mask)
    o = np.array([[0xbc00a82a, 0xbc00007d, 0x7d00ff00, 0x00000000, 0x00000000],
                  [0xbc00007d, 0xbc00a82a, 0x7d00007d, 0x00000000, 0x00000000],
                  [0x7d00ff00, 0x7d00007d, 0xbca85600, 0x00000000, 0x7dff0000],
                  [0x00000000, 0x00000000, 0x00000000, 0x7dff0000, 0x00000000],
                  [0x00000000, 0x00000000, 0x7dff0000, 0x00000000, 0x7dff0000]])
    np.testing.assert_equal(s.data, o)

    s = tf.spread(img, px=0)
    np.testing.assert_equal(s.data, img.data)

    pytest.raises(ValueError, lambda: tf.spread(img, px=-1))
    pytest.raises(ValueError, lambda: tf.spread(img, mask=np.ones(2)))
    pytest.raises(ValueError, lambda: tf.spread(img, mask=np.ones((2, 2))))


def test_uint32_spread():
    data = np.array([[1, 1, 0, 0, 0],
                     [1, 2, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 3, 0],
                     [0, 0, 0, 0, 0]], dtype='uint32')
    coords = [np.arange(5), np.arange(5)]
    arr = xr.DataArray(data, coords=coords, dims=dims)

    s = tf.spread(arr)
    o = np.array([[5, 5, 3, 0, 0],
                  [5, 5, 3, 0, 0],
                  [3, 3, 5, 3, 3],
                  [0, 0, 3, 3, 3],
                  [0, 0, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)
    assert (s.x_axis == arr.x_axis).all()
    assert (s.y_axis == arr.y_axis).all()
    assert s.dims == arr.dims

    s = tf.spread(arr, px=2)
    o = np.array([[5, 5, 5, 3, 0],
                  [5, 5, 8, 6, 3],
                  [5, 8, 7, 5, 3],
                  [3, 6, 5, 3, 3],
                  [0, 3, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)

    s = tf.spread(arr, shape='square')
    o = np.array([[5, 5, 3, 0, 0],
                  [5, 5, 3, 0, 0],
                  [3, 3, 5, 3, 3],
                  [0, 0, 3, 3, 3],
                  [0, 0, 3, 3, 3]])

    np.testing.assert_equal(s.data, o)

    s = tf.spread(arr, how='min')
    o = np.array([[1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 2, 3, 3],
                  [0, 0, 3, 3, 3],
                  [0, 0, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)

    s = tf.spread(arr, how='max')

    o = np.array([[2, 2, 2, 0, 0],
                  [2, 2, 2, 0, 0],
                  [2, 2, 3, 3, 3],
                  [0, 0, 3, 3, 3],
                  [0, 0, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)


    mask = np.array([[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]])

    data = np.array([[0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]], dtype='uint32')
    arr = xr.DataArray(data, coords=coords, dims=dims)
    s = tf.spread(arr, mask=mask)

    o = np.array([[0, 0, 0, 1, 0],
                  [1, 0, 2, 0, 1],
                  [0, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0]])
    np.testing.assert_equal(s.data, o)

    s = tf.spread(arr, px=0)
    np.testing.assert_equal(s.data, arr.data)

    pytest.raises(ValueError, lambda: tf.spread(arr, px=-1))
    pytest.raises(ValueError, lambda: tf.spread(arr, mask=np.ones(2)))
    pytest.raises(ValueError, lambda: tf.spread(arr, mask=np.ones((2, 2))))


def test_int32_spread():
    data = np.array([[1, 1, 0, 0, 0],
                     [1, 2, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 3, 0],
                     [0, 0, 0, 0, 0]], dtype='int32')
    coords = [np.arange(5), np.arange(5)]
    arr = xr.DataArray(data, coords=coords, dims=dims)

    s = tf.spread(arr)
    o = np.array([[5, 5, 3, 0, 0],
                  [5, 5, 3, 0, 0],
                  [3, 3, 5, 3, 3],
                  [0, 0, 3, 3, 3],
                  [0, 0, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)
    assert (s.x_axis == arr.x_axis).all()
    assert (s.y_axis == arr.y_axis).all()
    assert s.dims == arr.dims

    s = tf.spread(arr, px=2)
    o = np.array([[5, 5, 5, 3, 0],
                  [5, 5, 8, 6, 3],
                  [5, 8, 7, 5, 3],
                  [3, 6, 5, 3, 3],
                  [0, 3, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)

    s = tf.spread(arr, shape='square')
    o = np.array([[5, 5, 3, 0, 0],
                  [5, 5, 3, 0, 0],
                  [3, 3, 5, 3, 3],
                  [0, 0, 3, 3, 3],
                  [0, 0, 3, 3, 3]])

    np.testing.assert_equal(s.data, o)

    s = tf.spread(arr, how='min')
    o = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    np.testing.assert_equal(s.data, o)

    s = tf.spread(arr, how='max')

    o = np.array([[2, 2, 2, 0, 0],
                  [2, 2, 2, 0, 0],
                  [2, 2, 3, 3, 3],
                  [0, 0, 3, 3, 3],
                  [0, 0, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)


    mask = np.array([[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]])

    data = np.array([[0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]], dtype='int32')
    arr = xr.DataArray(data, coords=coords, dims=dims)
    s = tf.spread(arr, mask=mask)

    o = np.array([[0, 0, 0, 1, 0],
                  [1, 0, 2, 0, 1],
                  [0, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0]])
    np.testing.assert_equal(s.data, o)

    s = tf.spread(arr, px=0)
    np.testing.assert_equal(s.data, arr.data)

    pytest.raises(ValueError, lambda: tf.spread(arr, px=-1))
    pytest.raises(ValueError, lambda: tf.spread(arr, mask=np.ones(2)))
    pytest.raises(ValueError, lambda: tf.spread(arr, mask=np.ones((2, 2))))


def test_float32_spread():
    data = np.array([[1, 1, np.nan, np.nan, np.nan],
                     [1, 2, np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.nan, 3, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan]], dtype='float32')
    coords = [np.arange(5), np.arange(5)]
    arr = xr.DataArray(data, coords=coords, dims=dims)

    s = tf.spread(arr)
    o = np.array([[5, 5, 3, np.nan, np.nan],
                  [5, 5, 3, np.nan, np.nan],
                  [3, 3, 5, 3, 3],
                  [np.nan, np.nan, 3, 3, 3],
                  [np.nan, np.nan, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)
    assert (s.x_axis == arr.x_axis).all()
    assert (s.y_axis == arr.y_axis).all()
    assert s.dims == arr.dims

    s = tf.spread(arr, px=2)
    o = np.array([[5, 5, 5, 3, np.nan],
                  [5, 5, 8, 6, 3],
                  [5, 8, 7, 5, 3],
                  [3, 6, 5, 3, 3],
                  [np.nan, 3, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)

    s = tf.spread(arr, shape='square')
    o = np.array([[5, 5, 3, np.nan, np.nan],
                  [5, 5, 3, np.nan, np.nan],
                  [3, 3, 5, 3, 3],
                  [np.nan, np.nan, 3, 3, 3],
                  [np.nan, np.nan, 3, 3, 3]])

    np.testing.assert_equal(s.data, o)

    s = tf.spread(arr, how='min')
    o = np.array([[1, 1, 1, np.nan, np.nan],
                  [1, 1, 1, np.nan, np.nan],
                  [1, 1, 2, 3, 3],
                  [np.nan, np.nan, 3, 3, 3],
                  [np.nan, np.nan, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)

    s = tf.spread(arr, how='max')

    o = np.array([[2, 2, 2, np.nan, np.nan],
                  [2, 2, 2, np.nan, np.nan],
                  [2, 2, 3, 3, 3],
                  [np.nan, np.nan, 3, 3, 3],
                  [np.nan, np.nan, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)


    mask = np.array([[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]])
    data = np.array([[np.nan, np.nan, np.nan, 1, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan],
                     [np.nan, 1, np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan]], dtype='float32')
    arr = xr.DataArray(data, coords=coords, dims=dims)
    s = tf.spread(arr, mask=mask)


    o = np.array([[0, 0, 0, 1, 0],
                  [1, 0, 2, 0, 1],
                  [0, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0]])

    o = np.array([[np.nan, np.nan, np.nan, 1, np.nan],
                  [1, np.nan, 2, np.nan, 1],
                  [np.nan, 1, np.nan, np.nan, np.nan],
                  [1, np.nan, 1, np.nan, np.nan],
                  [np.nan, np.nan, np.nan, np.nan, np.nan]])
    np.testing.assert_equal(s.data, o)

    s = tf.spread(arr, px=0)
    np.testing.assert_equal(s.data, arr.data)

    pytest.raises(ValueError, lambda: tf.spread(arr, px=-1))
    pytest.raises(ValueError, lambda: tf.spread(arr, mask=np.ones(2)))
    pytest.raises(ValueError, lambda: tf.spread(arr, mask=np.ones((2, 2))))


def test_categorical_spread():
    a_data = np.array([[0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]], dtype='int32')

    b_data = np.array([[0, 0, 0, 0, 0],
                       [0, 2, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]], dtype='int32')

    c_data = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 3, 0],
                       [0, 0, 0, 0, 0]], dtype='int32')

    data = np.dstack([a_data, b_data, c_data])
    coords = [np.arange(5), np.arange(5)]
    arr = xr.DataArray(data, coords=coords + [['a', 'b', 'c']],
                       dims=dims + ['cat'])

    s = tf.spread(arr)
    o = np.array([[1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    np.testing.assert_equal(s.sel(cat='a').data, o)

    o = np.array([[2, 2, 2, 0, 0],
                  [2, 2, 2, 0, 0],
                  [2, 2, 2, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    np.testing.assert_equal(s.sel(cat='b').data, o)

    o = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 3, 3, 3],
                  [0, 0, 3, 3, 3],
                  [0, 0, 3, 3, 3]])
    np.testing.assert_equal(s.sel(cat='c').data, o)


def test_rgb_density():
    b = 0xffff0000
    data = np.full((4, 4), b, dtype='uint32')
    assert tf._rgb_density(data) == 1.0
    data = np.zeros((4, 4), dtype='uint32')
    assert tf._rgb_density(data) == np.inf
    data[3, 3] = b
    assert tf._rgb_density(data) == 0
    data[2, 0] = data[0, 2] = data[1, 1] = b
    assert np.allclose(tf._rgb_density(data), 0.75)
    assert np.allclose(tf._rgb_density(data, 3), 1)

def test_int_array_density():
    data = np.ones((4, 4), dtype='uint32')
    assert tf._array_density(data, float_type=False) == 1.0
    data = np.zeros((4, 4), dtype='uint32')
    assert tf._array_density(data, float_type=False) == np.inf
    data[3, 3] = 1
    assert tf._array_density(data, float_type=False) == 0
    data[2, 0] = data[0, 2] = data[1, 1] = 1
    assert np.allclose(tf._array_density(data, float_type=False), 0.75)
    assert np.allclose(tf._array_density(data, float_type=False, px=3), 1)


def test_float_array_density():
    data = np.ones((4, 4), dtype='float32')
    assert tf._array_density(data, float_type=True) == 1.0
    data = np.full((4, 4), np.nan, dtype='float32')
    assert tf._array_density(data, float_type=True) == np.inf
    data[3, 3] = 1
    assert tf._array_density(data, float_type=True) == 0
    data[2, 0] = data[0, 2] = data[1, 1] = 1
    assert np.allclose(tf._array_density(data, float_type=True), 0.75)
    assert np.allclose(tf._array_density(data, float_type=True, px=3), 1)


def test_rgb_dynspread():
    b = 0xffff0000
    coords = [np.arange(5), np.arange(5)]
    data = np.array([[b, b, 0, 0, 0],
                     [b, b, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, b, 0],
                     [0, 0, 0, 0, 0]], dtype='uint32')
    img = tf.Image(data, coords=coords, dims=dims)
    assert tf.dynspread(img).equals(img)
    data = np.array([[b, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [b, 0, 0, 0, b],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]], dtype='uint32')
    img = tf.Image(data, coords=coords, dims=dims)
    assert tf.dynspread(img, threshold=0.4).equals(tf.spread(img, 0))
    assert tf.dynspread(img, threshold=0.7).equals(tf.spread(img, 1))
    assert tf.dynspread(img, threshold=1.0).equals(tf.spread(img, 3))
    assert tf.dynspread(img, max_px=0).equals(img)

    pytest.raises(ValueError, lambda: tf.dynspread(img, threshold=1.1))
    pytest.raises(ValueError, lambda: tf.dynspread(img, max_px=-1))

def test_array_dynspread():
    coords = [np.arange(5), np.arange(5)]
    data = np.array([[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0]], dtype='uint32')
    arr = xr.DataArray(data, coords=coords, dims=dims)
    assert tf.dynspread(arr).equals(arr)
    data = np.array([[1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]], dtype='uint32')
    arr = xr.DataArray(data, coords=coords, dims=dims)
    assert tf.dynspread(arr, threshold=0.4).equals(tf.spread(arr, 0))
    assert tf.dynspread(arr, threshold=0.7).equals(tf.spread(arr, 1))
    assert tf.dynspread(arr, threshold=1.0).equals(tf.spread(arr, 3))
    assert tf.dynspread(arr, max_px=0).equals(arr)

    pytest.raises(ValueError, lambda: tf.dynspread(arr, threshold=1.1))
    pytest.raises(ValueError, lambda: tf.dynspread(arr, max_px=-1))


def test_categorical_dynspread():
    a_data = np.array([[1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]], dtype='int32')

    b_data = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]], dtype='int32')

    c_data = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]], dtype='int32')

    data = np.dstack([a_data, b_data, c_data])
    coords = [np.arange(5), np.arange(5)]
    arr = xr.DataArray(data, coords=coords + [['a', 'b', 'c']],
                       dims=dims + ['cat'])
    assert tf.dynspread(arr, threshold=0.4).equals(tf.spread(arr, 0))
    assert tf.dynspread(arr, threshold=0.7).equals(tf.spread(arr, 1))
    assert tf.dynspread(arr, threshold=1.0).equals(tf.spread(arr, 3))
    assert tf.dynspread(arr, max_px=0).equals(arr)


def check_eq_hist_cdf_slope(eq):
    # Check that the slope of the cdf is ~1
    # Adapted from scikit-image's test for the same function
    cdf = np.histogram(eq[~np.isnan(eq)], bins=256)[0].cumsum()
    cdf = cdf / cdf[-1]
    slope = np.polyfit(np.linspace(0, 1, cdf.size), cdf, 1)[0]
    assert 0.9 < slope < 1.1


def test_eq_hist(rng):
    # Float
    data = rng.normal(size=300**2)
    data[rng.integers(300**2, size=100)] = np.nan
    data = (data - np.nanmin(data)).reshape((300, 300))
    mask = np.isnan(data)
    eq, _ = tf.eq_hist(data, mask)
    check_eq_hist_cdf_slope(eq)
    assert (np.isnan(eq) == mask).all()
    # Integer
    data = rng.normal(scale=100, size=(300, 300)).astype('i8')
    data = data - data.min()
    eq, _ = tf.eq_hist(data)
    check_eq_hist_cdf_slope(eq)


def test_Image_to_pil():
    PIL = pytest.importorskip('PIL')
    img = img1.to_pil()
    assert isinstance(img, PIL.Image.Image)


def test_Image_to_bytesio():
    pytest.importorskip('PIL')

    bytes = img1.to_bytesio()
    assert isinstance(bytes, BytesIO)
    assert bytes.tell() == 0


def test_shade_should_handle_zeros_array():
    data = np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]], dtype='uint32')
    arr = tf.Image(data, dims=['x', 'y'])
    img = tf.shade(arr, cmap=['white', 'black'], how='linear')
    assert img is not None


def test_shade_with_discrete_color_key(array):
    data = array([[0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 0],
                     [0, 2, 2, 2, 0],
                     [0, 3, 3, 3, 0],
                     [0, 0, 0, 0, 0]], dtype='uint32')
    color_key = {1: 'white', 2: 'purple', 3: 'yellow'}
    result = array([[0, 0, 0, 0, 0],
                       [0, 4294967295, 4294967295, 4294967295, 0],
                       [0, 4286578816, 4286578816, 4286578816, 0],
                       [0, 4278255615, 4278255615, 4278255615, 0],
                       [0, 0, 0, 0, 0]],
                      dtype='uint32')

    arr = tf.Image(data, dims=['x', 'y'])
    result = tf.shade(arr, color_key=color_key)
    assert (result.data == result).all()


def test_interpolate_alpha_discrete_levels_None(array, request):
    if "dask" in request.node.name:
        pytest.skip("This test is not compatible with dask arrays")
    data = array([[0.0, 1.0], [1.0, 0.0]])
    # Issue #1084: this raises a ValueError.
    tf._interpolate_alpha(data, data, None, "eq_hist", 0.5, None, 0.4, True)
