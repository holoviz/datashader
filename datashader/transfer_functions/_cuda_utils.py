from __future__ import division

from distutils.version import LooseVersion
from math import ceil, isnan

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
if LooseVersion(numba.__version__) >= LooseVersion("0.51.0"):
    @cuda.jit(device=True)
    def cuda_atomic_nanmin(ary, idx, val):
        return cuda.atomic.nanmin(ary, idx, val)
    @cuda.jit(device=True)
    def cuda_atomic_nanmax(ary, idx, val):
        return cuda.atomic.nanmax(ary, idx, val)
elif LooseVersion(numba.__version__) <= LooseVersion("0.49.1"):
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


# interp
# ------
# When cupy adds cupy.interp support, this function can be removed
def interp(x, xp, fp, left=None, right=None):
    """
    cupy implementation of np.interp.  This function can be removed when an
    official cupy.interp function is added to the cupy library.
    """
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
