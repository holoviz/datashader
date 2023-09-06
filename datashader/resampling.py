"""
This module was adapted from https://github.com/CAB-LAB/gridtools

                        The MIT License (MIT)

Copyright (c) 2016, Brockmann Consult GmbH and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from itertools import groupby
from math import floor, ceil

import dask.array as da
import numpy as np

from dask.delayed import delayed
from numba import prange
from .utils import ngjit, ngjit_parallel

try:
    import cupy
except Exception:
    cupy = None


#: Interpolation method for upsampling: Take nearest source grid cell, even if it is invalid.
US_NEAREST = 10
#: Interpolation method for upsampling: Bi-linear interpolation between the 4 nearest source grid cells.
US_LINEAR = 11

#: Aggregation method for downsampling: Take first valid source grid cell, ignore contribution areas.
DS_FIRST = 50
#: Aggregation method for downsampling: Take last valid source grid cell, ignore contribution areas.
DS_LAST = 51
#: Aggregation method for downsampling: Take the minimum source grid cell value, ignore contribution areas.
DS_MIN = 52
#: Aggregation method for downsampling: Take the maximum source grid cell value, ignore contribution areas.
DS_MAX = 53
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

#: Constant indicating an empty 2-D mask
_NOMASK2D = np.ma.getmaskarray(np.ma.array([[0]], mask=[[0]]))

_EPS = 1e-10

upsample_methods   = dict(nearest=US_NEAREST, linear=US_LINEAR)

downsample_methods = dict(first=DS_FIRST, last=DS_LAST, mode=DS_MODE,
                          mean=DS_MEAN,   var=DS_VAR,   std=DS_STD,
                          min=DS_MIN,     max=DS_MAX)


def map_chunks(in_shape, out_shape, out_chunks):
    """
    Maps index in source array to target array chunks.

    For each chunk in the target array this function computes the
    indexes into the source array that will be fed into the regridding
    operation.

    Parameters
    ----------
    in_shape: tuple(int, int)
      The shape of the input array
    out_shape: tuple(int, int)
      The shape of the output array
    out_chunks: tuple(int, int)
      The shape of each chunk in the output array

    Returns
    -------
      Dictionary mapping of chunks and their indexes
      in the input and output array.
    """
    outy, outx = out_shape
    cys, cxs = out_chunks
    xchunks = list(range(0, outx, cxs)) + [outx]
    ychunks = list(range(0, outy, cys)) + [outy]
    iny, inx = in_shape
    xscale = inx/outx
    yscale = iny/outy
    mapping = {}
    for i in range(len(ychunks)-1):
        cumy0, cumy1 = ychunks[i:i+2]
        iny0, iny1 = cumy0*yscale, cumy1*yscale
        iny0r, iny1r = floor(iny0), ceil(iny1)
        y0_off, y1_off = iny0-iny0r, iny1r-iny1
        for j in range(len(xchunks)-1):
            cumx0, cumx1 = xchunks[j:j+2]
            inx0, inx1 = cumx0*xscale, cumx1*xscale
            inx0r, inx1r = floor(inx0), ceil(inx1)
            x0_off, x1_off = inx0-inx0r, inx1r-inx1
            mapping[(i, j)] = {
                'out': {
                    'x': (cumx0, cumx1),
                    'y': (cumy0, cumy1),
                    'w': (cumx1-cumx0),
                    'h': (cumy1-cumy0),
                },
                'in': {
                    'x': (inx0r, inx1r),
                    'y': (iny0r, iny1r),
                    'xoffset': (x0_off, x1_off),
                    'yoffset': (y0_off, y1_off),
                }
            }
    return mapping


def compute_chunksize(src, w, h, chunksize=None, max_mem=None):
    """
    Attempts to compute a chunksize for the resampling output array
    that is as close as possible to the input array chunksize, while
    also respecting the maximum memory constraint to avoid loading
    to much data into memory at the same time.

    Parameters
    ----------
    src : dask.array.Array
        The source array to resample
    w : int
        New grid width
    h : int
        New grid height
    chunksize : tuple(int, int) (optional)
        Size of the output chunks. By default the chunk size is
        inherited from the *src* array.
    max_mem : int (optional)
        The maximum number of bytes that should be loaded into memory
        during the regridding operation.

    Returns
    -------
    chunksize : tuple(int, int)
        Size of the output chunks.
    """
    start_chunksize = src.chunksize if chunksize is None else chunksize
    if max_mem is None:
        return start_chunksize

    sh, sw = src.shape
    height_fraction = float(sh)/h
    width_fraction = float(sw)/w
    ch, cw = start_chunksize
    dim = True
    nbytes = src.dtype.itemsize
    while ((ch * height_fraction) * (cw * width_fraction) * nbytes) > max_mem:
        if dim:
            cw -= 1
        else:
            ch -= 1
        dim = not dim
    if ch == 0 or cw == 0:
        min_mem = height_fraction * width_fraction * nbytes
        raise ValueError(
            "Given the memory constraints the resampling operation "
            "could not find a chunksize that avoids loading too much "
            "data into memory. Either relax the memory constraint to "
            "a minimum of %d bytes or resample to a larger grid size. "
            "Note: A future implementation could handle this condition "
            "by declaring temporary arrays." % min_mem)
    return ch, cw


def resample_2d_distributed(src, w, h, ds_method='mean', us_method='linear',
                            fill_value=None, mode_rank=1, chunksize=None,
                            max_mem=None):
    """
    A distributed version of 2-d grid resampling which operates on
    dask arrays and performs regridding on a chunked array.

    Parameters
    ----------
    src : dask.array.Array
        The source array to resample
    w : int
        New grid width
    h : int
        New grid height
    ds_method : str (optional)
        Grid cell aggregation method for a possible downsampling
        (one of the *DS_* constants).
    us_method : str (optional)
        Grid cell interpolation method for a possible upsampling
        (one of the *US_* constants, optional).
    fill_value : scalar (optional)
        If None, numpy's default value is used.
    mode_rank : scalar (optional)
        The rank of the frequency determined by the *ds_method*
        ``DS_MODE``. One (the default) means most frequent value, two
        means second most frequent value, and so forth.
    chunksize : tuple(int, int) (optional)
        Size of the output chunks. By default this the chunk size is
        inherited from the *src* array.
    max_mem : int (optional)
        The maximum number of bytes that should be loaded into memory
        during the regridding operation.

    Returns
    -------
    resampled : dask.array.Array
        A resampled version of the *src* array.
    """
    temp_chunks = compute_chunksize(src, w, h, chunksize, max_mem)
    if chunksize is None:
        chunksize = src.chunksize

    chunk_map = map_chunks(src.shape, (h, w), temp_chunks)
    out_chunks = {}
    for (i, j), chunk in chunk_map.items():
        inds = chunk['in']
        inx0, inx1 = inds['x']
        iny0, iny1 = inds['y']
        out = chunk['out']
        chunk_array = src[iny0:iny1, inx0:inx1]
        resampled = _resample_2d_delayed(
                chunk_array, out['w'], out['h'], ds_method, us_method,
                fill_value, mode_rank, inds['xoffset'], inds['yoffset'])
        out_chunks[(i, j)] = {
            'array': resampled,
            'shape': (out['h'], out['w']),
            'dtype': src.dtype,
            'in': chunk['in'],
            'out': out
        }

    rows = groupby(out_chunks.items(), lambda x: x[0][0])
    cols = []
    for i, row in rows:
        row = da.concatenate([
            da.from_delayed(chunk['array'], chunk['shape'], chunk['dtype'])
            for _, chunk in row], 1)
        cols.append(row)
    out = da.concatenate(cols, 0)

    # Ensure chunksize conforms to specified chunksize
    if chunksize is not None and out.chunksize != chunksize:
        out = out.rechunk(chunksize)
    return out


def resample_2d(src, w, h, ds_method='mean', us_method='linear',
                fill_value=None, mode_rank=1, x_offset=(0, 0),
                y_offset=(0, 0), out=None):
    """
    Resample a 2-D grid to a new resolution.

    Parameters
    ----------
    src : np.ndarray
        The source array to resample
    w : int
        New grid width
    h : int
        New grid height
    ds_method : str (optional)
        Grid cell aggregation method for a possible downsampling
        (one of the *DS_* constants).
    us_method : str (optional)
        Grid cell interpolation method for a possible upsampling
        (one of the *US_* constants, optional).
    fill_value : scalar (optional)
        If ``None``, it is taken from **src** if it is a masked array,
        otherwise from *out* if it is a masked array,
        otherwise numpy's default value is used.
    mode_rank : scalar (optional)
        The rank of the frequency determined by the *ds_method*
        ``DS_MODE``. One (the default) means most frequent value, zwo
        means second most frequent value, and so forth.
    x_offset : tuple(float, float) (optional)
        Offsets for the x-axis indices in the source array (useful
        for distributed regridding where chunks are not aligned with
        the underlying array).
    y_offset : tuple(float, float) (optional)
        Offsets for the x-axis indices in the source array (useful
        for distributed regridding where chunks are not aligned with
        the underlying array).
    out : numpy.ndarray (optional)
        Alternate output array in which to place the result. The
        default is *None*; if provided, it must have the same shape as
        the expected output.

    Returns
    -------
    resampled : numpy.ndarray or dask.array.Array
        A resampled version of the *src* array.
    """
    out = _get_out(out, src, (h, w))
    if out is None:
        return src
    mask, use_mask = _get_mask(src)
    fill_value = _get_fill_value(fill_value, src, out)

    us_method=upsample_methods[us_method]
    ds_method=downsample_methods[ds_method]

    if isinstance(src, np.ma.MaskedArray):
        src = src.data

    resampled = _resample_2d(src, mask, use_mask, ds_method, us_method,
                             fill_value, mode_rank, x_offset, y_offset, out)
    return _mask_or_not(resampled, src, fill_value)


_resample_2d_delayed = delayed(resample_2d)


def upsample_2d(src, w, h, method=US_LINEAR, fill_value=None, out=None):
    """
    Upsample a 2-D grid to a higher resolution by interpolating original grid cells.

    src: 2-D *ndarray*
    w: *int*
        Grid width, which must be greater than or equal to *src.shape[-1]*
    h:  *int*
        Grid height, which must be greater than or equal to *src.shape[-2]*
    method: one of the *US_* constants, optional
        Grid cell interpolation method
    fill_value: *scalar*, optional
        If ``None``, it is taken from **src** if it is a masked array,
        otherwise from *out* if it is a masked array,
        otherwise numpy's default value is used.
    out: 2-D *ndarray*, optional
        Alternate output array in which to place the result. The default is *None*; if provided, it must have the same
        shape as the expected output.

    Returns
    -------
    upsampled : numpy.ndarray or dask.array.Array
        An upsampled version of the *src* array.
    """
    out = _get_out(out, src, (h, w))
    if out is None:
        return src
    mask, use_mask = _get_mask(src)
    fill_value = _get_fill_value(fill_value, src, out)

    if method not in UPSAMPLING_METHODS:
        raise ValueError('invalid upsampling method')

    upsampling_method = UPSAMPLING_METHODS[method]
    upsampled = upsampling_method(
        src, mask, use_mask, fill_value, (0, 0), (0, 0), out)
    return _mask_or_not(upsampled, src, fill_value)


def downsample_2d(src, w, h, method=DS_MEAN, fill_value=None, mode_rank=1, out=None):
    """
    Downsample a 2-D grid to a lower resolution by aggregating original grid cells.

    Parameters
    ----------
    src : numpy.ndarray or dask.array.Array
        The source array to resample
    w : int
        New grid width
    h : int
        New grid height
    ds_method : str (optional)
        Grid cell aggregation method for a possible downsampling
        (one of the *DS_* constants).
    fill_value : scalar (optional)
        If ``None``, it is taken from **src** if it is a masked array,
        otherwise from *out* if it is a masked array,
        otherwise numpy's default value is used.
    mode_rank : scalar (optional)
        The rank of the frequency determined by the *ds_method*
        ``DS_MODE``. One (the default) means most frequent value, two
        means second most frequent value, and so forth.
    out : numpy.ndarray (optional)
        Alternate output array in which to place the result. The
        default is *None*; if provided, it must have the same shape as
        the expected output.

    Returns
    -------
    downsampled : numpy.ndarray or dask.array.Array
        An downsampled version of the *src* array.
    """
    if method == DS_MODE and mode_rank < 1:
        raise ValueError('mode_rank must be >= 1')
    out = _get_out(out, src, (h, w))
    if out is None:
        return src
    mask, use_mask = _get_mask(src)
    fill_value = _get_fill_value(fill_value, src, out)

    if method not in DOWNSAMPLING_METHODS:
        raise ValueError('invalid downsampling method')

    downsampling_method = DOWNSAMPLING_METHODS[method]
    downsampled = downsampling_method(
        src, mask, use_mask, method, fill_value, mode_rank, (0, 0),
        (0, 0), out)
    return _mask_or_not(downsampled, src, fill_value)


def _get_out(out, src, shape):
    if out is None:
        return np.zeros(shape, dtype=src.dtype)
    else:
        if out.shape != shape:
            raise ValueError("'shape' and 'out' are incompatible")
        if out.shape == src.shape:
            return None
        return out


def _get_mask(src):
    if isinstance(src, np.ma.MaskedArray):
        mask = np.ma.getmask(src)
        if mask is not np.ma.nomask:
            return mask, True
    return _NOMASK2D, False


def _mask_or_not(out, src, fill_value):
    if isinstance(src, np.ma.MaskedArray):
        if not isinstance(out, np.ma.MaskedArray):
            if np.isfinite(fill_value):
                masked = np.ma.masked_equal(out, fill_value, copy=False)
            else:
                masked = np.ma.masked_invalid(out, copy=False)
            masked.set_fill_value(fill_value)
            return masked
    return out


def _get_fill_value(fill_value, src, out):
    if fill_value is None:
        if isinstance(src, np.ma.MaskedArray):
            fill_value = src.fill_value
        elif isinstance(out, np.ma.MaskedArray):
            fill_value = out.fill_value
        else:
            # use numpy's default fill_value
            fill_value = np.ma.array([0], mask=[False], dtype=src.dtype).fill_value
    return fill_value


@ngjit
def _get_dimensions(src, out):
    src_w = src.shape[-1]
    src_h = src.shape[-2]
    out_w = out.shape[-1]
    out_h = out.shape[-2]
    return src_w, src_h, out_w, out_h


def _resample_2d(src, mask, use_mask, ds_method, us_method, fill_value,
                 mode_rank, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)
    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    src_wo = (src_w - x0_off - x1_off)
    src_ho = (src_h - y0_off - y1_off)

    if us_method not in UPSAMPLING_METHODS:
        raise ValueError('invalid upsampling method')
    elif ds_method not in DOWNSAMPLING_METHODS:
        raise ValueError('invalid downsampling method')

    downsampling_method = DOWNSAMPLING_METHODS[ds_method]
    upsampling_method = UPSAMPLING_METHODS[us_method]

    if src_h == 0 or src_w == 0 or out_h == 0 or out_w == 0:
       return np.zeros((out_h, out_w), dtype=src.dtype)
    elif out_w < src_wo and out_h < src_ho:
        return downsampling_method(src, mask, use_mask, ds_method,
                                   fill_value, mode_rank, x_offset,
                                   y_offset, out)
    elif out_w < src_wo:
        if out_h > src_ho:
            temp = np.zeros((src_h, out_w), dtype=src.dtype)
            temp = downsampling_method(src, mask, use_mask, ds_method,
                                       fill_value, mode_rank, x_offset,
                                       y_offset,  temp)
            # todo - write test & fix: must use mask=np.ma.getmaskarray(temp) here if use_mask==True
            return upsampling_method(temp, mask, use_mask, fill_value,
                                     x_offset, y_offset, out)
        else:
            return downsampling_method(src, mask, use_mask, ds_method,
                                       fill_value, mode_rank, x_offset,
                                       y_offset, out)
    elif out_h < src_ho:
        if out_w > src_wo:
            temp = np.zeros((out_h, src_w), dtype=src.dtype)
            temp = downsampling_method(src, mask, use_mask, ds_method,
                                       fill_value, mode_rank, x_offset,
                                       y_offset,  temp)
            # todo - write test & fix: must use mask=np.ma.getmaskarray(temp) here if use_mask==True
            return upsampling_method(temp, mask, use_mask, fill_value,
                                     x_offset, y_offset, out)
        else:
            return downsampling_method(src, mask, use_mask, ds_method,
                                       fill_value, mode_rank, x_offset,
                                       y_offset, out)
    elif out_w > src_wo or out_h > src_ho:
        return upsampling_method(src, mask, use_mask, fill_value,
                                 x_offset, y_offset,  out)
    return src


@ngjit_parallel
def _upsample_2d_nearest(src, mask, use_mask, fill_value, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)
    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    src_w = (src_w - x0_off - x1_off)
    src_h = (src_h - y0_off - y1_off)

    if src_w == out_w and src_h == out_h:
        return src

    if out_w < src_w or out_h < src_h:
        raise ValueError("invalid target size")

    scale_x = src_w / out_w
    scale_y = src_h / out_h

    for out_y in prange(out_h):
        src_y = int((scale_y * out_y) + y0_off)
        for out_x in range(out_w):
            src_x = int((scale_x * out_x) + x0_off)
            value = src[src_y, src_x]
            if np.isfinite(value) and not (use_mask and mask[src_y, src_x]):
                out[out_y, out_x] = value
            else:
                out[out_y, out_x] = fill_value
    return out


@ngjit_parallel
def _upsample_2d_linear(src, mask, use_mask, fill_value, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)
    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    src_wo = (src_w - x0_off - x1_off)
    src_ho = (src_h - y0_off - y1_off)

    if src_wo == out_w and src_ho == out_h:
        return src

    if out_w < src_w or out_h < src_h:
        raise ValueError("invalid target size")

    scale_x = (src_wo - 1.0) / ((out_w - 1.0) if out_w > 1 else 1.0)
    scale_y = (src_ho - 1.0) / ((out_h - 1.0) if out_h > 1 else 1.0)
    for out_y in prange(out_h):
        src_yf = (scale_y * out_y) + y0_off
        src_y0 = int(src_yf)
        wy = src_yf - src_y0
        src_y1 = src_y0 + 1
        if src_y1 >= src_h:
            src_y1 = src_y0
        for out_x in range(out_w):
            src_xf = (scale_x * out_x) + x0_off
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
    return out


UPSAMPLING_METHODS = {US_LINEAR: _upsample_2d_linear,
                      US_NEAREST: _upsample_2d_nearest}


@ngjit_parallel
def _downsample_2d_first_last(src, mask, use_mask, method, fill_value,
                              mode_rank, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)

    if src_w == out_w and src_h == out_h:
        return src

    if out_w > src_w or out_h > src_h:
        raise ValueError("invalid target size")

    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    scale_x = (src_w - x0_off - x1_off) / out_w
    scale_y = (src_h - y0_off - y1_off) / out_h

    for out_y in prange(out_h):
        src_yf0 = (scale_y * out_y) + y0_off
        src_yf1 = src_yf0 + scale_y
        src_y0 = int(src_yf0)
        src_y1 = int(src_yf1)
        wy1 = src_yf1 - src_y1
        if wy1 < _EPS and src_y1 > src_y0:
            src_y1 -= 1
        for out_x in range(out_w):
            src_xf0 = (scale_x * out_x) + x0_off
            src_xf1 = src_xf0 + scale_x
            src_x0 = int(src_xf0)
            src_x1 = int(src_xf1)
            wx1 = src_xf1 - src_x1
            if wx1 < _EPS and src_x1 > src_x0:
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
    return out


@ngjit_parallel
def _downsample_2d_min_max(src, mask, use_mask, method, fill_value,
                           mode_rank, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)

    if src_w == out_w and src_h == out_h:
        return src

    if out_w > src_w or out_h > src_h:
        raise ValueError("invalid target size")

    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    scale_x = (src_w - x0_off - x1_off) / out_w
    scale_y = (src_h - y0_off - y1_off) / out_h

    for out_y in prange(out_h):
        src_yf0 = (scale_y * out_y) + y0_off
        src_yf1 = src_yf0 + scale_y
        src_y0 = int(src_yf0)
        src_y1 = int(src_yf1)
        wy1 = src_yf1 - src_y1
        if wy1 < _EPS and src_y1 > src_y0:
            src_y1 -= 1
        for out_x in range(out_w):
            src_xf0 = (scale_x * out_x) + x0_off
            src_xf1 = src_xf0 + scale_x
            src_x0 = int(src_xf0)
            src_x1 = int(src_xf1)
            wx1 = src_xf1 - src_x1
            if wx1 < _EPS and src_x1 > src_x0:
                src_x1 -= 1
            if method == DS_MIN:
                value = np.inf
            else:
                value = -np.inf
            for src_y in range(src_y0, src_y1 + 1):
                for src_x in range(src_x0, src_x1 + 1):
                    v = src[src_y, src_x]
                    if np.isfinite(v) and not (use_mask and mask[src_y, src_x]):
                        if method == DS_MIN:
                            if v < value:
                                value = v
                        else:
                            if v > value:
                                value = v
            if np.isfinite(value):
                out[out_y, out_x] = value
            else:
                out[out_y, out_x] = fill_value
    return out


@ngjit_parallel
def _downsample_2d_mode(src, mask, use_mask, method, fill_value,
                        mode_rank, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)

    if src_w == out_w and src_h == out_h:
        return src

    if out_w > src_w or out_h > src_h:
        raise ValueError("invalid target size")

    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    scale_x = (src_w - x0_off - x1_off) / out_w
    scale_y = (src_h - y0_off - y1_off) / out_h

    max_value_count = ceil(scale_x + 1) * ceil(scale_y + 1)
    if mode_rank >= max_value_count:
        raise ValueError("requested mode_rank too large for max_value_count being collected")

    for out_y in prange(out_h):
        src_yf0 = (scale_y * out_y) + y0_off
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
            values = np.zeros((max_value_count,), dtype=src.dtype)
            frequencies = np.zeros((max_value_count,), dtype=np.uint32)

            src_xf0 = (scale_x * out_x) + x0_off
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
            if mode_rank == 1:
                for i in range(value_count):
                    w = frequencies[i]
                    if w > w_max:
                        w_max = w
                        value = values[i]
            elif mode_rank <= max_value_count:
                max_frequencies = np.full(mode_rank, -1.0, dtype=np.float64)
                indices = np.zeros(mode_rank, dtype=np.int64)
                for i in range(value_count):
                    w = frequencies[i]
                    for j in range(mode_rank):
                        if w > max_frequencies[j]:
                            max_frequencies[j] = w
                            indices[j] = i
                            break
                value = values[indices[mode_rank - 1]]
            out[out_y, out_x] = value
    return out


@ngjit_parallel
def _downsample_2d_mean(src, mask, use_mask, method, fill_value,
                        mode_rank, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)

    if src_w == out_w and src_h == out_h:
        return src

    if out_w > src_w or out_h > src_h:
        raise ValueError("invalid target size")

    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    scale_x = (src_w - x0_off - x1_off) / out_w
    scale_y = (src_h - y0_off - y1_off) / out_h

    for out_y in prange(out_h):
        src_yf0 = (scale_y * out_y) + y0_off
        src_yf1 = (src_yf0 + scale_y)
        src_y0 = int(src_yf0)
        src_y1 = int(src_yf1)

        wy0 = 1.0 - (src_yf0 - src_y0)
        wy1 = src_yf1 - src_y1
        if wy1 < _EPS:
            wy1 = 1.0
            if src_y1 > src_y0:
                src_y1 -= 1
        for out_x in range(out_w):
            src_xf0 = (scale_x * out_x) + x0_off
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
    return out


@ngjit_parallel
def _downsample_2d_std_var(src, mask, use_mask, method, fill_value,
                           mode_rank, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)

    if src_w == out_w and src_h == out_h:
        return src

    if out_w > src_w or out_h > src_h:
        raise ValueError("invalid target size")

    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    scale_x = (src_w - x0_off - x1_off) / out_w
    scale_y = (src_h - y0_off - y1_off) / out_h

    for out_y in prange(out_h):
        src_yf0 = (scale_y * out_y) + y0_off
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
            src_xf0 = (scale_x * out_x) + x0_off
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
    return out


DOWNSAMPLING_METHODS = {DS_MEAN: _downsample_2d_mean,
                        DS_FIRST: _downsample_2d_first_last,
                        DS_LAST: _downsample_2d_first_last,
                        DS_MIN: _downsample_2d_min_max,
                        DS_MAX: _downsample_2d_min_max,
                        DS_MODE: _downsample_2d_mode,
                        DS_STD: _downsample_2d_std_var,
                        DS_VAR: _downsample_2d_std_var}


def infer_interval_breaks(coord, axis=0):
    """
    >>> infer_interval_breaks(np.arange(5))
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
    >>> infer_interval_breaks([[0, 1], [3, 4]], axis=1)
    array([[-0.5,  0.5,  1.5],
           [ 2.5,  3.5,  4.5]])
    """
    if cupy and isinstance(coord, cupy.ndarray):
        # leave cupy array as-is
        pass
    else:
        coord = np.asarray(coord)
    if len(coord) == 0:
        return np.array([], dtype=coord.dtype)
    deltas = 0.5 * np.diff(coord, axis=axis)
    first = np.take(coord, [0], axis=axis) - np.take(deltas, [0], axis=axis)
    last = np.take(coord, [-1], axis=axis) + np.take(deltas, [-1], axis=axis)
    trim_last = tuple(slice(None, -1) if n == axis else slice(None)
                      for n in range(coord.ndim))
    return np.concatenate([first, coord[trim_last] + deltas, last], axis=axis)
