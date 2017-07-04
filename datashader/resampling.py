from __future__ import absolute_import, division, print_function

import numpy as np
import param
from .utils import ngjit

#: Constant indicating an empty 2-D mask
_NOMASK2D = np.ma.getmaskarray(np.ma.array([[0]], mask=[[0]]))

_EPS = 1e-10

# TODO: tidy docstrings

# wrappers around _upsample_2d()
class UpsampleMethod(param.ParameterizedFunction):
    __abstract = True
    method = None
    def __call__(self,src,mask,use_mask,fill_value,out,**params):
        #p = param.ParamOverrides(params)
        return _upsample_2d(src,mask,use_mask,self.method,fill_value,out)

class upsample_nearest(UpsampleMethod):
    """Take nearest source grid cell, even if it is
          invalid."""
    # TODO: figure out what can be passed to jit'd fn 
    #method = b"nearest" # TODO python 2
    method = 1

class upsample_linear(UpsampleMethod):
    """Bi-linear interpolation between the 4 nearest
          source grid cells."""
    #method = b"linear"
    method = 2


# wrappers around _downsample_2d()
class DownsampleMethod(param.ParameterizedFunction):
    __abstract = True
    method = None
    def __call__(self,src,mask,use_mask,fill_value,out,**params):
        p = param.ParamOverrides(self, params)
        return _downsample_2d(src,mask,use_mask,self.method,fill_value,out,**p)

class downsample_first(DownsampleMethod):
    """Take first valid source grid cell, ignore contribution areas."""
    #method = b"first"
    method = 10

class downsample_last(DownsampleMethod):
    """Take last valid source grid cell, ignore contribution areas."""
    #method = b"last"
    method = 11
    
class downsample_mean(DownsampleMethod):
    """
    Compute average of all valid source grid cells,
    with weights given by contribution area.
    """
    method = 12
    #method = b"mean"
        
class downsample_mode(DownsampleMethod):
    """
    Compute most frequently seen valid source grid cell, with
    frequency given by contribution area.
    """
    #method = b"mode"
    method = 13
    
    rank = param.Integer(default=1,bounds=(0,None),doc="""
    can be used to generate the n-th mode
    or
    The rank of the frequency determined by the *ds_method* ``DS_MODE``. One (the default) means
    most frequent value, zwo means second most frequent value, and so forth.""")
    
class downsample_var(DownsampleMethod):
    """
    Compute the biased weighted estimator of variance (see
    https://en.wikipedia.org/wiki/Mean_square_weighted_deviation),
    with weights given by contribution area.
    """
    #method = b"var"
    method = 14
    
class downsample_std(DownsampleMethod):
    """
    Compute the corresponding standard deviation to the biased
    weighted estimator of variance (see
    https://en.wikipedia.org/wiki/Mean_square_weighted_deviation),
    with weights given by contribution area.
    """
    #method = b"std"
    method = 15


# TODO: PR doc dropping upsample, downsample interfaces (but could
# restore)

class resample_2d(param.ParameterizedFunction):
    """
    Resample a 2-D grid to a new resolution.
    """
    # TODO: or did I mean to use instances?
    ds_method = param.ClassSelector(DownsampleMethod,default=downsample_mean,is_instance=False,doc="""
        grid cell aggregation method for a possible downsampling.""")

    us_method = param.ClassSelector(UpsampleMethod,default=upsample_linear,is_instance=False,doc="""
        Grid cell interpolation method for a possible upsampling""")

    # :param fill_value: *scalar*, optional
    fill_value = param.Number(default=None, doc="""
        If ``None``, it is taken from **src** if it is a masked array otherwise from *out* if it is a masked array, otherwise numpy's default value is used.""")
    
    # TODO: WIP, you are here - shrinking this down to put in __call__
    @staticmethod
    def _resample_2d(src, mask, use_mask, ds_method, us_method, fill_value, out):
        src_w = src.shape[-1]
        src_h = src.shape[-2]
        out_w = out.shape[-1]
        out_h = out.shape[-2]
        
        if out_w < src_w and out_h < src_h:
            return ds_method(src, mask, use_mask, fill_value, out)
        elif out_w < src_w:
            if out_h > src_h:
                temp = np.zeros((src_h, out_w), dtype=src.dtype)
                temp = ds_method(src, mask, use_mask, fill_value, temp)
                # todo - write test & fix: must use mask=np.ma.getmaskarray(temp) here if use_mask==True
                return us_method(temp, mask, use_mask, fill_value, out)
            else:
                return ds_method(src, mask, use_mask, fill_value, out)
        elif out_h < src_h:
            if out_w > src_w:
                temp = np.zeros((out_h, src_w), dtype=src.dtype)
                temp = ds_method(src, mask, use_mask, fill_value, temp)
                # todo - write test & fix: must use mask=np.ma.getmaskarray(temp) here if use_mask==True
                return us_method(temp, mask, use_mask, fill_value, out)
            else:
                return ds_method(src, mask, use_mask, fill_value, out)
        elif out_w > src_w or out_h > src_h:
            return us_method(src, mask, use_mask, fill_value, out)
        return src
        

    def __call__(self, src, w, h, out=None, **params):
        """
        :param src: 2-D *ndarray*
        :param w: *int*
            New grid width
        :param h:  *int*
            New grid height
        :param out: 2-D *ndarray*, optional
            Alternate output array in which to place the result. The default is *None*; if provided, it must have the same
            shape as the expected output.

        :return: An resampled version of the *src* array.
        """
        p = param.ParamOverrides(self,params)
        
        out = _get_out(out, src, (h, w))
        if out is None:
            return src
        mask, use_mask = _get_mask(src)
        fill_value = _get_fill_value(p.fill_value, src, out)
        
        return _mask_or_not(
            self._resample_2d(src, mask, use_mask, p.ds_method, p.us_method, fill_value, out),
            src, fill_value)
    


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
def _upsample_2d(src, mask, use_mask, method, fill_value, out):
    src_w = src.shape[-1]
    src_h = src.shape[-2]
    out_w = out.shape[-1]
    out_h = out.shape[-2]

    if src_w == out_w and src_h == out_h:
        return src

    if out_w < src_w or out_h < src_h:
        raise ValueError("invalid target size")

    #if method == upsample_nearest.method:
    if method == 1:
    #if method == b"nearest":
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

    #elif method == upsample_linear.method:
    elif method == 2:
    #elif method == b"linear":
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

    
# TODO: haven't dealt with rank yet; default of 1 should not be left.

@ngjit
def _downsample_2d(src, mask, use_mask, method, fill_value, out, rank=1):
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

    #if method == b"first" or method==b"last":
    if method == 10 or method==11:
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
                            #if method == b"first":
                            if method == 10:    
                                done = True
                                break
                    if done:
                        break
                out[out_y, out_x] = value

    elif method == 13: #b"mode":
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

    #elif method == b"mean":
    elif method == 12:        
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

    #elif method == b"var" or method == b"std":
    elif method == 14 or method == 15:        
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
        #if method == b"std":
        if method == 15:                       
            out = np.sqrt(out)
    else:
        raise ValueError('invalid upsampling method')

    return out
