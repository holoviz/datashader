from __future__ import annotations

from math import ceil, isnan
from packaging.version import Version

try:
    from math import nan
except:
    nan = float('nan')

import numba
import numpy as np

from numba import cuda

try:
    import cupy
    if cupy.result_type is np.result_type:
        # Workaround until cupy release of https://github.com/cupy/cupy/pull/2249
        # Without this, cupy.histogram raises an error that cupy.result_type
        # is not defined.
        cupy.result_type = lambda *args: np.result_type(
            *[arg.dtype if isinstance(arg, cupy.ndarray) else arg
              for arg in args]
        )
except:
    cupy = None


def cuda_args(shape):
    """
    Compute the blocks-per-grid and threads-per-block parameters for use when
    invoking cuda kernels

    Parameters
    ----------
    shape: int or tuple of ints
        The shape of the input array that the kernel will parallelize over

    Returns
    -------
    tuple
        Tuple of (blocks_per_grid, threads_per_block)
    """
    if isinstance(shape, int):
        shape = (shape,)

    max_threads = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    # Note: We divide max_threads by 2.0 to leave room for the registers
    # occupied by the kernel. For some discussion, see
    # https://github.com/numba/numba/issues/3798.
    threads_per_block = int(ceil(max_threads / 2.0) ** (1.0 / len(shape)))
    tpb = (threads_per_block,) * len(shape)
    bpg = tuple(int(ceil(d / threads_per_block)) for d in shape)
    return bpg, tpb


# masked_clip_2d
# --------------
def masked_clip_2d(data, mask, lower, upper):
    """
    Clip the elements of an input array between lower and upper bounds,
    skipping over elements that are masked out.

    Parameters
    ----------
    data: cupy.ndarray
        Numeric ndarray that will be clipped in-place
    mask: cupy.ndarray
        Boolean ndarray where True values indicate elements that should be
        skipped
    lower: int or float
        Lower bound to clip to
    upper: int or float
        Upper bound to clip to

    Returns
    -------
    None
        data array is modified in-place
    """
    masked_clip_2d_kernel[cuda_args(data.shape)](data, mask, lower, upper)


# Behaviour of numba.cuda.atomic.max/min changed in 0.50 so as to behave as per
# np.nanmax/np.nanmin
if Version(numba.__version__) >= Version("0.51.0"):
    @cuda.jit(device=True)
    def cuda_atomic_nanmin(ary, idx, val):
        return cuda.atomic.nanmin(ary, idx, val)
    @cuda.jit(device=True)
    def cuda_atomic_nanmax(ary, idx, val):
        return cuda.atomic.nanmax(ary, idx, val)
elif Version(numba.__version__) <= Version("0.49.1"):
    @cuda.jit(device=True)
    def cuda_atomic_nanmin(ary, idx, val):
        return cuda.atomic.min(ary, idx, val)
    @cuda.jit(device=True)
    def cuda_atomic_nanmax(ary, idx, val):
        return cuda.atomic.max(ary, idx, val)
else:
    raise ImportError("Datashader's CUDA support requires numba!=0.50.0")


@cuda.jit
def masked_clip_2d_kernel(data, mask, lower, upper):
    i, j = cuda.grid(2)
    maxi, maxj = data.shape
    if i >= 0 and i < maxi and j >= 0 and j < maxj and not mask[i, j]:
        cuda_atomic_nanmax(data, (i, j), lower)
        cuda_atomic_nanmin(data, (i, j), upper)


def interp(x, xp, fp, left=None, right=None):
    """
    cupy implementation of np.interp, falls back to cupy implementation
    if available.
    """
    x = cupy.asarray(x)
    xp = cupy.asarray(xp)
    fp = cupy.asarray(fp)
    if hasattr(cupy, 'interp'):
        return cupy.interp(x, xp, fp, left, right)
    output_y = cupy.zeros(x.shape, dtype=cupy.float64)
    assert len(x.shape) == 2
    if left is None:
        left = fp[0]
    left = float(left)
    if right is None:
        right = fp[-1]
    right = float(right)
    interp2d_kernel[cuda_args(x.shape)](
        x.astype(cupy.float64), xp.astype(cupy.float64), fp.astype(cupy.float64), left, right, output_y
    )
    return output_y


@cuda.jit
def interp2d_kernel(x, xp, fp, left, right, output_y):
    i, j = cuda.grid(2)
    if i < x.shape[0] and j < x.shape[1]:
        xval = x[i, j]

        if isnan(xval):
            output_y[i, j] = nan
        elif xval < xp[0]:
            output_y[i, j] = left
        elif xval >= xp[-1]:
            output_y[i, j] = right
        else:
            # Find indices of xp that straddle xval
            upper_i = len(xp) - 1
            lower_i = 0
            while True:
                stop_i = 1 + (lower_i + upper_i) // 2
                if xp[stop_i] < xval:
                    lower_i = stop_i
                elif xp[stop_i - 1] > xval:
                    upper_i = stop_i - 1
                else:
                    break

            # Compute interpolate y value
            x0 = xp[stop_i - 1]
            x1 = xp[stop_i]
            y0 = fp[stop_i - 1]
            y1 = fp[stop_i]

            slope = (y1 - y0) / (x1 - x0)
            y_interp = y0 + slope * (xval - x0)

            # Update output
            output_y[i, j] = y_interp


if Version(numba.__version__) >= Version("0.57"):
    # See issues #1196 and #1211.
    @cuda.jit(device=True)
    def cuda_mutex_lock(mutex, index):
        while cuda.atomic.cas(mutex, index, 0, 1) != 0:
            pass
        cuda.threadfence()

    @cuda.jit(device=True)
    def cuda_mutex_unlock(mutex, index):
        cuda.threadfence()
        cuda.atomic.exch(mutex, index, 0)
else:
    @cuda.jit(device=True)
    def cuda_mutex_lock(mutex, index):
        while cuda.atomic.compare_and_swap(mutex, 0, 1) != 0:
              pass
        cuda.threadfence()

    @cuda.jit(device=True)
    def cuda_mutex_unlock(mutex, index):
        cuda.threadfence()
        cuda.atomic.exch(mutex, 0, 0)


@cuda.jit(device=True)
def cuda_shift_and_insert(target, value, index):
    """Insert a value into a 1D array at a particular index, but before doing
    that shift the previous values along one to make room. For use in
    ``FloatingNReduction`` classes such as ``max_n`` and ``first_n`` which
    store ``n`` values per pixel.

    Parameters
    ----------
    target : 1d numpy array
        Target pixel array.

    value : float
        Value to insert into target pixel array.

    index : int
        Index to insert at.

    Returns
    -------
    Index beyond insertion, i.e. where the first shifted value now sits.
    """
    n = len(target)
    for i in range(n-1, index, -1):
        target[i] = target[i-1]
    target[index] = value
    return index + 1


@cuda.jit(device=True)
def _cuda_nanmax_n_impl(ret_pixel, other_pixel):
    """Single pixel implementation of nanmax_n_in_place.
    ret_pixel and other_pixel are both 1D arrays of the same length.

    Walk along other_pixel a value at a time, find insertion index in
    ret_pixel and shift values along to insert.  Next other_pixel value is
    inserted at a higher index, so this walks the two pixel arrays just once
    each.
    """
    n = len(ret_pixel)
    istart = 0
    for other_value in other_pixel:
        if isnan(other_value):
            break
        else:
            for i in range(istart, n):
                if isnan(ret_pixel[i]) or other_value > ret_pixel[i]:
                    istart = cuda_shift_and_insert(ret_pixel, other_value, i)
                    break


@cuda.jit
def cuda_nanmax_n_in_place_4d(ret, other):
    """CUDA equivalent of nanmax_n_in_place_4d.
    """
    ny, nx, ncat, _n = ret.shape
    x, y, cat = cuda.grid(3)
    if x < nx and y < ny and cat < ncat:
        _cuda_nanmax_n_impl(ret[y, x, cat], other[y, x, cat])


@cuda.jit
def cuda_nanmax_n_in_place_3d(ret, other):
    """CUDA equivalent of nanmax_n_in_place_3d.
    """
    ny, nx, _n = ret.shape
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        _cuda_nanmax_n_impl(ret[y, x], other[y, x])


@cuda.jit(device=True)
def _cuda_nanmin_n_impl(ret_pixel, other_pixel):
    """Single pixel implementation of nanmin_n_in_place.
    ret_pixel and other_pixel are both 1D arrays of the same length.

    Walk along other_pixel a value at a time, find insertion index in
    ret_pixel and shift values along to insert.  Next other_pixel value is
    inserted at a higher index, so this walks the two pixel arrays just once
    each.
    """
    n = len(ret_pixel)
    istart = 0
    for other_value in other_pixel:
        if isnan(other_value):
            break
        else:
            for i in range(istart, n):
                if isnan(ret_pixel[i]) or other_value < ret_pixel[i]:
                    istart = cuda_shift_and_insert(ret_pixel, other_value, i)
                    break


@cuda.jit
def cuda_nanmin_n_in_place_4d(ret, other):
    """CUDA equivalent of nanmin_n_in_place_4d.
    """
    ny, nx, ncat, _n = ret.shape
    x, y, cat = cuda.grid(3)
    if x < nx and y < ny and cat < ncat:
        _cuda_nanmin_n_impl(ret[y, x, cat], other[y, x, cat])


@cuda.jit
def cuda_nanmin_n_in_place_3d(ret, other):
    """CUDA equivalent of nanmin_n_in_place_3d.
    """
    ny, nx, _n = ret.shape
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        _cuda_nanmin_n_impl(ret[y, x], other[y, x])


@cuda.jit
def cuda_row_min_in_place(ret, other):
    """CUDA equivalent of row_min_in_place.
    """
    ny, nx, ncat = ret.shape
    x, y, cat = cuda.grid(3)
    if x < nx and y < ny and cat < ncat:
        if other[y, x, cat] > -1 and (ret[y, x, cat] == -1 or other[y, x, cat] < ret[y, x, cat]):
            ret[y, x, cat] = other[y, x, cat]


@cuda.jit(device=True)
def _cuda_row_max_n_impl(ret_pixel, other_pixel):
    """Single pixel implementation of row_max_n_in_place.
    ret_pixel and other_pixel are both 1D arrays of the same length.

    Walk along other_pixel a value at a time, find insertion index in
    ret_pixel and shift values along to insert.  Next other_pixel value is
    inserted at a higher index, so this walks the two pixel arrays just once
    each.
    """
    n = len(ret_pixel)
    istart = 0
    for other_value in other_pixel:
        if other_value == -1:
            break
        else:
            for i in range(istart, n):
                if ret_pixel[i] == -1 or other_value > ret_pixel[i]:
                    istart = cuda_shift_and_insert(ret_pixel, other_value, i)
                    break


@cuda.jit
def cuda_row_max_n_in_place_4d(ret, other):
    """CUDA equivalent of row_max_n_in_place_4d.
    """
    ny, nx, ncat, _n = ret.shape
    x, y, cat = cuda.grid(3)
    if x < nx and y < ny and cat < ncat:
        _cuda_row_max_n_impl(ret[y, x, cat], other[y, x, cat])


@cuda.jit
def cuda_row_max_n_in_place_3d(ret, other):
    """CUDA equivalent of row_max_n_in_place_3d.
    """
    ny, nx, _n = ret.shape
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        _cuda_row_max_n_impl(ret[y, x], other[y, x])


@cuda.jit(device=True)
def _cuda_row_min_n_impl(ret_pixel, other_pixel):
    """Single pixel implementation of row_min_n_in_place.
    ret_pixel and other_pixel are both 1D arrays of the same length.

    Walk along other_pixel a value at a time, find insertion index in
    ret_pixel and shift values along to insert.  Next other_pixel value is
    inserted at a higher index, so this walks the two pixel arrays just once
    each.
    """
    n = len(ret_pixel)
    istart = 0
    for other_value in other_pixel:
        if other_value == -1:
            break
        else:
            for i in range(istart, n):
                if ret_pixel[i] == -1 or other_value < ret_pixel[i]:
                    istart = cuda_shift_and_insert(ret_pixel, other_value, i)
                    break


@cuda.jit
def cuda_row_min_n_in_place_4d(ret, other):
    """CUDA equivalent of row_min_n_in_place_4d.
    """
    ny, nx, ncat, _n = ret.shape
    x, y, cat = cuda.grid(3)
    if x < nx and y < ny and cat < ncat:
        _cuda_row_min_n_impl(ret[y, x, cat], other[y, x, cat])


@cuda.jit
def cuda_row_min_n_in_place_3d(ret, other):
    """CUDA equivalent of row_min_n_in_place_4=3d.
    """
    ny, nx, _n = ret.shape
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        _cuda_row_min_n_impl(ret[y, x], other[y, x])
