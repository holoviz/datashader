"""
resample_2d(): resample 2D grid to a new resolution with options
for up and downsampling (specified by Upsample and Downsample
family of functions).
"""

from __future__ import absolute_import, division

import numpy as np
import param
from ._resampling import _upsample_2d, _downsample_2d

#: Constant indicating an empty 2-D mask
_NOMASK2D = np.ma.getmaskarray(np.ma.array([[0]], mask=[[0]]))

class Sample2D(param.ParameterizedFunction):
    """
    Abstract base class for wrappers around underlying jit'd
    _upsample_2d() and _downsample_2d().

    Concrete subclasses should specify the _sampler and _method class
    attribute to control the up/down sampling method employed (see
    _up/downsample_2d).
    """
    __abstract = True
    _sampler = None
    _method = None    
    
    def __call__(self,src,mask,use_mask,fill_value,out,**params):
        assert self._method is not None
        assert self._sampler is not None
        p = param.ParamOverrides(self, params)
        # allow this fn to pass all its params to _sampler fn
        # TODO: verify it's worked - are there actually any tests for
        # the different sampling methods?
        ##
        # hack to be sorted out in param then removed from here:
        # paramoverrides.get_param_values, basically
        _x = dict(p._overridden.get_param_values()) 
        _x.update(p)
        del _x['name']
        ##
        return self._sampler(src,mask,use_mask,self._method,fill_value,out,**_x)


class Upsample(Sample2D):
    __abstract = True
    _sampler = staticmethod(_upsample_2d)
    
class upsample_nearest(Upsample):
    """Take nearest source grid cell, even if it is invalid."""
    _method = 10

class upsample_linear(Upsample):
    """Bi-linear interpolation between the 4 nearest source grid cells."""
    _method = 11


class Downsample(Sample2D):
    __abstract = True
    _sampler = staticmethod(_downsample_2d)

class downsample_first(Downsample):
    """Take first valid source grid cell, ignore contribution areas."""
    _method = 50

class downsample_last(Downsample):
    """Take last valid source grid cell, ignore contribution areas."""
    _method = 51
    
class downsample_mean(Downsample):
    """
    Compute average of all valid source grid cells,
    with weights given by contribution area.
    """
    _method = 54
        
class downsample_mode(Downsample):
    """
    Compute most frequently seen valid source grid cell, with
    frequency given by contribution area.
    """
    _method = 56
    
    rank = param.Integer(default=1,bounds=(0,None),doc="""
        The rank of the frequency. One (the default) means most frequent
        value, zwo means second most frequent value, and so forth.""")
    
class downsample_var(Downsample):
    """
    Compute the biased weighted estimator of variance (see
    https://en.wikipedia.org/wiki/Mean_square_weighted_deviation),
    with weights given by contribution area.
    """
    _method = 57
    
class downsample_std(Downsample):
    """
    Compute the corresponding standard deviation to the biased
    weighted estimator of variance (see
    https://en.wikipedia.org/wiki/Mean_square_weighted_deviation),
    with weights given by contribution area.
    """
    _method = 58


class resample_2d(param.ParameterizedFunction):
    """
    Resample a 2-D grid to a new resolution.
    """
    # TODO: rename to downsample_method or downsampler or something? (Same for us_method)
    
    # TODO: or did I mean to use instances?
    ds_method = param.ClassSelector(Downsample,default=downsample_mean,is_instance=False,doc="""
        grid cell aggregation method for a possible downsampling.""")

    us_method = param.ClassSelector(Upsample,default=upsample_linear,is_instance=False,doc="""
        Grid cell interpolation method for a possible upsampling""")

    fill_value = param.Number(default=None, doc="""
        If ``None``, it is taken from **src** if it is a masked array
        otherwise from *out* if it is a masked array, otherwise numpy's
        default value is used.""")
    
    def __call__(self, src, w, h, out=None, **params):
        """
        :param src: 2-D *ndarray*
        :param w: *int*
            New grid width
        :param h:  *int*
            New grid height
        :param out: 2-D *ndarray*, optional
            Alternate output array in which to place the result. The
            default is *None*; if provided, it must have the same
            shape as the expected output.
        :return: An resampled version of the *src* array.
        """
        p = param.ParamOverrides(self,params)
        
        out = _get_out(out, src, (h, w))
        if out is None:
            return src
        
        mask, use_mask = _get_mask(src)
        fill_value = _get_fill_value(p.fill_value, src, out)

        src_w = src.shape[-1]
        src_h = src.shape[-2]
        out_w = out.shape[-1]
        out_h = out.shape[-2]

        # TODO: would be nice to simplify this...
        if out_w < src_w and out_h < src_h:
            resampled = p.ds_method(src, mask, use_mask, fill_value, out)
        elif out_w < src_w:
            if out_h > src_h:
                temp = np.zeros((src_h, out_w), dtype=src.dtype)
                temp = p.ds_method(src, mask, use_mask, fill_value, temp)
                # todo - write test & fix: must use mask=np.ma.getmaskarray(temp) here if use_mask==True
                resampled = p.us_method(temp, mask, use_mask, fill_value, out)
            else:
                resampled = p.ds_method(src, mask, use_mask, fill_value, out)
        elif out_h < src_h:
            if out_w > src_w:
                temp = np.zeros((out_h, src_w), dtype=src.dtype)
                temp = p.ds_method(src, mask, use_mask, fill_value, temp)
                # todo - write test & fix: must use mask=np.ma.getmaskarray(temp) here if use_mask==True
                resampled = p.us_method(temp, mask, use_mask, fill_value, out)
            else:
                resampled = p.ds_method(src, mask, use_mask, fill_value, out)
        elif out_w > src_w or out_h > src_h:
            resampled = p.us_method(src, mask, use_mask, fill_value, out)
        else:
            resampled = src
 
        return _mask_or_not(resampled, src, fill_value)
    


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
    
