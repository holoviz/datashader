"""
Functions to provide resampling capabilities; see resampling.py for
interface and documentation.
"""
from __future__ import absolute_import, division

# Differences from gridtools.resampling:
#
# * removed upsample_2d, downsample_2d, resample_2d interface fns
#
# * removed _resample_2d (why was that jit'd?) - should test
#   performance now
#
# * ngjit rather than jit
#
# * swapped parameters of _downsample_2d() around to allow rank to be
#   optional and added assertion that rank >= 1 in _downsample_2d (and
#   renamed mode_rank to rank)
#
# * _get_out(), _get_mask(), _mask_or_not(), _get_fill_value() moved
#   out to resampling.py

import numpy as np

from .utils import ngjit

#: Interpolation method for upsampling: Take nearest source grid cell, even if it is invalid.
US_NEAREST = 10
#: Interpolation method for upsampling: Bi-linear interpolation between the 4 nearest source grid cells.
US_LINEAR = 11

#: Aggregation method for downsampling: Take first valid source grid cell, ignore contribution areas.
DS_FIRST = 50
#: Aggregation method for downsampling: Take last valid source grid cell, ignore contribution areas.
DS_LAST = 51
# DS_MIN = 52
# DS_MAX = 53
#: Aggregation method for downsampling: Compute average of all valid source grid cells,
#: with weights given by contribution area.
DS_MEAN = 54
# DS_MEDIAN = 55
#: Aggregation method for downsampling: Compute most frequently seen valid source grid cell,
#: with frequency given by contribution area. Note that this mode can use an additional keyword argument
#: *mode_rank* which can be used to generate the n-th mode. See :py:function:`downsample_2d`.
DS_MODE = 56
#: Aggregation method for downsampling: Compute the biased weighted estimator of variance
#: (see https://en.wikipedia.org/wiki/Mean_square_weighted_deviation), with weights given by contribution area.
DS_VAR = 57
#: Aggregation method for downsampling: Compute the corresponding standard deviation to the biased weighted estimator
#: of variance
#: (see https://en.wikipedia.org/wiki/Mean_square_weighted_deviation), with weights given by contribution area.
DS_STD = 58


_EPS = 1e-10

@ngjit
def _upsample_2d(src, mask, use_mask, method, fill_value, out):
    src_w = src.shape[-1]
    src_h = src.shape[-2]
    out_w = out.shape[-1]
    out_h = out.shape[-2]

    if src_w == out_w and src_h == out_h:
        return src

    if out_w < src_w or out_h < src_h:
        raise ValueError("invalid target size")

    if method == US_NEAREST:
        scale_x = src_w / out_w
        scale_y = src_h / out_h
        for out_y in range(out_h):
            src_y = int(scale_y * out_y)
            for out_x in range(out_w):
                src_x = int(scale_x * out_x)
                value = src[src_y, src_x]
                if np.isfinite(value) and not (use_mask and mask[src_y, src_x]):
                    out[out_y, out_x] = value
                else:
                    out[out_y, out_x] = fill_value

    elif method == US_LINEAR:
        scale_x = (src_w - 1.0) / ((out_w - 1.0) if out_w > 1 else 1.0)
        scale_y = (src_h - 1.0) / ((out_h - 1.0) if out_h > 1 else 1.0)
        for out_y in range(out_h):
            src_yf = scale_y * out_y
            src_y0 = int(src_yf)
            wy = src_yf - src_y0
            src_y1 = src_y0 + 1
            if src_y1 >= src_h:
                src_y1 = src_y0
            for out_x in range(out_w):
                src_xf = scale_x * out_x
                src_x0 = int(src_xf)
                wx = src_xf - src_x0
                src_x1 = src_x0 + 1
                if src_x1 >= src_w:
                    src_x1 = src_x0
                v00 = src[src_y0, src_x0]
                v01 = src[src_y0, src_x1]
                v10 = src[src_y1, src_x0]
                v11 = src[src_y1, src_x1]
                if use_mask:
                    v00_ok = np.isfinite(v00) and not mask[src_y0, src_x0]
                    v01_ok = np.isfinite(v01) and not mask[src_y0, src_x1]
                    v10_ok = np.isfinite(v10) and not mask[src_y1, src_x0]
                    v11_ok = np.isfinite(v11) and not mask[src_y1, src_x1]
                else:
                    v00_ok = np.isfinite(v00)
                    v01_ok = np.isfinite(v01)
                    v10_ok = np.isfinite(v10)
                    v11_ok = np.isfinite(v11)
                if v00_ok and v01_ok and v10_ok and v11_ok:
                    ok = True
                    v0 = v00 + wx * (v01 - v00)
                    v1 = v10 + wx * (v11 - v10)
                    value = v0 + wy * (v1 - v0)
                elif wx < 0.5:
                    # NEAREST according to weight
                    if wy < 0.5:
                        ok = v00_ok
                        value = v00
                    else:
                        ok = v10_ok
                        value = v10
                else:
                    # NEAREST according to weight
                    if wy < 0.5:
                        ok = v01_ok
                        value = v01
                    else:
                        ok = v11_ok
                        value = v11
                if ok:
                    out[out_y, out_x] = value
                else:
                    out[out_y, out_x] = fill_value

    else:
        raise ValueError('invalid upsampling method')

    return out

@ngjit
def _downsample_2d(src, mask, use_mask, method, fill_value, out, rank=-1):
    src_w = src.shape[-1]
    src_h = src.shape[-2]
    out_w = out.shape[-1]
    out_h = out.shape[-2]

    if src_w == out_w and src_h == out_h:
        return src

    if out_w > src_w or out_h > src_h:
        raise ValueError("invalid target size")

    scale_x = src_w / out_w
    scale_y = src_h / out_h

    if method == DS_FIRST or method == DS_LAST:
        for out_y in range(out_h):
            src_yf0 = scale_y * out_y
            src_yf1 = src_yf0 + scale_y
            src_y0 = int(src_yf0)
            src_y1 = int(src_yf1)
            if src_y1 == src_yf1 and src_y1 > src_y0:
                src_y1 -= 1
            for out_x in range(out_w):
                src_xf0 = scale_x * out_x
                src_xf1 = src_xf0 + scale_x
                src_x0 = int(src_xf0)
                src_x1 = int(src_xf1)
                if src_x1 == src_xf1 and src_x1 > src_x0:
                    src_x1 -= 1
                done = False
                value = fill_value
                for src_y in range(src_y0, src_y1 + 1):
                    for src_x in range(src_x0, src_x1 + 1):
                        v = src[src_y, src_x]
                        if np.isfinite(v) and not (use_mask and mask[src_y, src_x]):
                            value = v
                            if method == DS_FIRST:    
                                done = True
                                break
                    if done:
                        break
                out[out_y, out_x] = value

    elif method == DS_MODE:
        if rank < 1:
            raise ValueError
        max_value_count = int(scale_x + 1) * int(scale_y + 1)
        values = np.zeros((max_value_count,), dtype=src.dtype)
        frequencies = np.zeros((max_value_count,), dtype=np.uint32)
        for out_y in range(out_h):
            src_yf0 = scale_y * out_y
            src_yf1 = src_yf0 + scale_y
            src_y0 = int(src_yf0)
            src_y1 = int(src_yf1)
            wy0 = 1.0 - (src_yf0 - src_y0)
            wy1 = src_yf1 - src_y1
            if wy1 < _EPS:
                wy1 = 1.0
                if src_y1 > src_y0:
                    src_y1 -= 1
            for out_x in range(out_w):
                src_xf0 = scale_x * out_x
                src_xf1 = src_xf0 + scale_x
                src_x0 = int(src_xf0)
                src_x1 = int(src_xf1)
                wx0 = 1.0 - (src_xf0 - src_x0)
                wx1 = src_xf1 - src_x1
                if wx1 < _EPS:
                    wx1 = 1.0
                    if src_x1 > src_x0:
                        src_x1 -= 1
                value_count = 0
                for src_y in range(src_y0, src_y1 + 1):
                    wy = wy0 if (src_y == src_y0) else wy1 if (src_y == src_y1) else 1.0
                    for src_x in range(src_x0, src_x1 + 1):
                        wx = wx0 if (src_x == src_x0) else wx1 if (src_x == src_x1) else 1.0
                        v = src[src_y, src_x]
                        if np.isfinite(v) and not (use_mask and mask[src_y, src_x]):
                            w = wx * wy
                            found = False
                            for i in range(value_count):
                                if v == values[i]:
                                    frequencies[i] += w
                                    found = True
                                    break
                            if not found:
                                values[value_count] = v
                                frequencies[value_count] = w
                                value_count += 1
                w_max = -1.
                value = fill_value
                if rank == 1:
                    for i in range(value_count):
                        w = frequencies[i]
                        if w > w_max:
                            w_max = w
                            value = values[i]
                elif rank <= max_value_count:
                    max_frequencies = np.full(rank, -1.0, dtype=np.float64)
                    indices = np.zeros(rank, dtype=np.int64)
                    for i in range(value_count):
                        w = frequencies[i]
                        for j in range(rank):
                            if w > max_frequencies[j]:
                                max_frequencies[j] = w
                                indices[j] = i
                                break
                    value = values[indices[rank - 1]]

                out[out_y, out_x] = value

    elif method == DS_MEAN:
        for out_y in range(out_h):
            src_yf0 = scale_y * out_y
            src_yf1 = src_yf0 + scale_y
            src_y0 = int(src_yf0)
            src_y1 = int(src_yf1)
            wy0 = 1.0 - (src_yf0 - src_y0)
            wy1 = src_yf1 - src_y1
            if wy1 < _EPS:
                wy1 = 1.0
                if src_y1 > src_y0:
                    src_y1 -= 1
            for out_x in range(out_w):
                src_xf0 = scale_x * out_x
                src_xf1 = src_xf0 + scale_x
                src_x0 = int(src_xf0)
                src_x1 = int(src_xf1)
                wx0 = 1.0 - (src_xf0 - src_x0)
                wx1 = src_xf1 - src_x1
                if wx1 < _EPS:
                    wx1 = 1.0
                    if src_x1 > src_x0:
                        src_x1 -= 1
                v_sum = 0.0
                w_sum = 0.0
                for src_y in range(src_y0, src_y1 + 1):
                    wy = wy0 if (src_y == src_y0) else wy1 if (src_y == src_y1) else 1.0
                    for src_x in range(src_x0, src_x1 + 1):
                        wx = wx0 if (src_x == src_x0) else wx1 if (src_x == src_x1) else 1.0
                        v = src[src_y, src_x]
                        if np.isfinite(v) and not (use_mask and mask[src_y, src_x]):
                            w = wx * wy
                            v_sum += w * v
                            w_sum += w
                if w_sum < _EPS:
                    out[out_y, out_x] = fill_value
                else:
                    out[out_y, out_x] = v_sum / w_sum

    elif method == DS_VAR or method == DS_STD:
        for out_y in range(out_h):
            src_yf0 = scale_y * out_y
            src_yf1 = src_yf0 + scale_y
            src_y0 = int(src_yf0)
            src_y1 = int(src_yf1)
            wy0 = 1.0 - (src_yf0 - src_y0)
            wy1 = src_yf1 - src_y1
            if wy1 < _EPS:
                wy1 = 1.0
                if src_y1 > src_y0:
                    src_y1 -= 1
            for out_x in range(out_w):
                src_xf0 = scale_x * out_x
                src_xf1 = src_xf0 + scale_x
                src_x0 = int(src_xf0)
                src_x1 = int(src_xf1)
                wx0 = 1.0 - (src_xf0 - src_x0)
                wx1 = src_xf1 - src_x1
                if wx1 < _EPS:
                    wx1 = 1.0
                    if src_x1 > src_x0:
                        src_x1 -= 1
                v_sum = 0.0
                w_sum = 0.0
                wv_sum = 0.0
                wvv_sum = 0.0
                for src_y in range(src_y0, src_y1 + 1):
                    wy = wy0 if (src_y == src_y0) else wy1 if (src_y == src_y1) else 1.0
                    for src_x in range(src_x0, src_x1 + 1):
                        wx = wx0 if (src_x == src_x0) else wx1 if (src_x == src_x1) else 1.0
                        v = src[src_y, src_x]
                        if np.isfinite(v) and not (use_mask and mask[src_y, src_x]):
                            w = wx * wy
                            v_sum += v
                            w_sum += w
                            wv_sum += w * v
                            wvv_sum += w * v * v
                if w_sum < _EPS:
                    out[out_y, out_x] = fill_value
                else:
                    out[out_y, out_x] = (wvv_sum * w_sum - wv_sum * wv_sum) / w_sum / w_sum
        if method == DS_STD:
            out = np.sqrt(out)
    else:
        raise ValueError('invalid upsampling method')

    return out




    
