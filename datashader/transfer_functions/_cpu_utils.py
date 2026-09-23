import numba as nb
import numpy as np

from datashader.utils import ngjit


@ngjit
def masked_clip_2d(data, mask, lower, upper):
    """
    Clip the elements of an input array between lower and upper bounds,
    skipping over elements that are masked out.

    Parameters
    ----------
    data: np.ndarray
        Numeric ndarray that will be clipped in-place
    mask: np.ndarray
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
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j]:
                continue
            val = data[i, j]
            if val < lower:
                data[i, j] = lower
            elif val > upper:
                data[i, j] = upper


@nb.jit(nogil=True, cache=True)
def interp_with_lut(x, xp, fp, lut, g0, inv_step):
    """``np.interp(x, xp, fp)`` for 1D float ``x``, bit-identical to NumPy.

    Instead of a binary search per element, the interval is guessed from a
    uniform grid (``lut[k]`` is the last ``xp`` index at or below grid point
    ``k``, where grid point ``k`` is at ``g0 + k / inv_step``) and then
    corrected with exact comparisons, so the guess only affects speed.
    The interpolation mirrors NumPy's implementation, including NaN handling.
    Requires ``len(xp) >= 2``.
    """
    n = len(xp)
    nlut = len(lut)
    out = np.empty(len(x), dtype=np.float64)
    for i in range(len(x)):
        xv = np.float64(x[i])
        if np.isnan(xv):
            out[i] = xv
            continue
        if xv < xp[0]:
            out[i] = fp[0]
            continue
        if xv > xp[n - 1]:
            out[i] = fp[n - 1]
            continue
        if xv == xp[n - 1]:
            out[i] = fp[n - 1]
            continue
        k = int((xv - g0) * inv_step)
        if k < 0:
            k = 0
        elif k >= nlut:
            k = nlut - 1
        j = lut[k]
        if j < 0:
            j = 0
        elif j > n - 2:
            j = n - 2
        while xv < xp[j]:
            j -= 1
        while xv >= xp[j + 1]:
            j += 1
        if xp[j] == xv:
            out[i] = fp[j]
            continue
        slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
        res = slope * (xv - xp[j]) + fp[j]
        if np.isnan(res):
            res = slope * (xv - xp[j + 1]) + fp[j + 1]
            if np.isnan(res) and fp[j] == fp[j + 1]:
                res = fp[j]
        out[i] = res
    return out
