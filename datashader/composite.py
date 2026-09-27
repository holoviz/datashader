"""
Binary graphical composition operators

See https://www.cairographics.org/operators/; more could easily be added from there.
"""

from __future__ import annotations

import numba as nb
import numpy as np
import os

image_operators = ('over', 'add', 'saturate', 'source')
array_operators = ('add_arr', 'max_arr', 'min_arr', 'source_arr')
__all__ = ('composite_op_lookup', 'validate_operator') + image_operators + array_operators


def validate_operator(how, is_image):
    name = how if is_image else how + '_arr'
    if is_image:
        if name not in image_operators:
            image_repr = ', '.join(repr(el) for el in image_operators)
            msg =f'Operator {how!r} not one of the supported image operators: {image_repr}'
            raise ValueError(msg)
    elif name not in array_operators:
        array_repr = ', '.join(repr(el[:-4]) for el in array_operators)
        msg = f'Operator {how!r} not one of the supported array operators: {array_repr}'
        raise ValueError(msg)


@nb.jit('(uint32,)', nopython=True, nogil=True, cache=True)
def extract_scaled(x):
    """Extract components as float64 values in [0.0, 1.0]"""
    r = np.float64(( x        & 255) / 255)
    g = np.float64(((x >>  8) & 255) / 255)
    b = np.float64(((x >> 16) & 255) / 255)
    a = np.float64(((x >> 24) & 255) / 255)
    return r, g, b, a


@nb.jit('(float64, float64, float64, float64)', nopython=True,
        nogil=True, cache=True)
def combine_scaled(r, g, b, a):
    """Combine components in [0, 1] to rgba uint32"""
    r2 = min(255, np.uint32(r * 255))
    g2 = min(255, np.uint32(g * 255))
    b2 = min(255, np.uint32(b * 255))
    a2 = min(255, np.uint32(a * 255))
    return np.uint32((a2 << 24) | (b2 << 16) | (g2 << 8) | r2)


jit_enabled = os.environ.get('NUMBA_DISABLE_JIT', '0') == '0'


if jit_enabled:
    extract_scaled.disable_compile()
    combine_scaled.disable_compile()


# Scalar kernels. The public operators of the same name are built from them below.
@nb.jit(nogil=True, cache=True)
def _source(src, dst):
    if src & 0xff000000:
        return src
    else:
        return dst


@nb.jit(nogil=True, cache=True)
def _over(src, dst):
    sr, sg, sb, sa = extract_scaled(src)
    dr, dg, db, da = extract_scaled(dst)

    factor = 1 - sa
    a = sa + da * factor
    if a == 0:
        return np.uint32(0)
    r = (sr * sa + dr * da * factor)/a
    g = (sg * sa + dg * da * factor)/a
    b = (sb * sa + db * da * factor)/a
    return combine_scaled(r, g, b, a)


@nb.jit(nogil=True, cache=True)
def _add(src, dst):
    sr, sg, sb, sa = extract_scaled(src)
    dr, dg, db, da = extract_scaled(dst)

    a = min(1, sa + da)
    if a == 0:
        return np.uint32(0)
    r = (sr * sa + dr * da)/a
    g = (sg * sa + dg * da)/a
    b = (sb * sa + db * da)/a
    return combine_scaled(r, g, b, a)


@nb.jit(nogil=True, cache=True)
def _saturate(src, dst):
    sr, sg, sb, sa = extract_scaled(src)
    dr, dg, db, da = extract_scaled(dst)

    a = min(1, sa + da)
    if a == 0:
        return np.uint32(0)
    factor = min(sa, 1 - da)
    r = (factor * sr + dr * da)/a
    g = (factor * sg + dg * da)/a
    b = (factor * sb + db * da)/a
    return combine_scaled(r, g, b, a)


@nb.jit(nogil=True, cache=True)
def _source_arr(src, dst):
    if src:
        return src
    else:
        return dst


@nb.jit(nogil=True, cache=True)
def _add_arr(src, dst):
    return src + dst


@nb.jit(nogil=True, cache=True)
def _max_arr(src, dst):
    return max(src, dst)


@nb.jit(nogil=True, cache=True)
def _min_arr(src, dst):
    return min(src, dst)


# Jitted code selects an operator by its index in `image_operators` or
# `array_operators`: a closure capturing a compiled function gets a different
# cache key in every process, but one capturing an int doesn't.
@nb.jit(nogil=True, cache=True, inline='always')
def _image_op(code, src, dst):
    if code == 0:
        return _over(src, dst)
    elif code == 1:
        return _add(src, dst)
    elif code == 2:
        return _saturate(src, dst)
    return _source(src, dst)


@nb.jit(nogil=True, cache=True, inline='always')
def _arr_op(code, src, dst):
    if code == 0:
        return _add_arr(src, dst)
    elif code == 1:
        return _max_arr(src, dst)
    elif code == 2:
        return _min_arr(src, dst)
    return _source_arr(src, dst)


def _loop_sigs(types, out_types=None):
    # Inputs are read-only broadcast views with any strides.
    inp = [nb.types.Array(t, 1, "A", readonly=True) for t in types]
    out = [nb.types.Array(t, 1, "C") for t in (out_types or types)]
    return [nb.void(i, i, o) for i, o in zip(inp, out)]


_ARR_TYPES = (nb.int32, nb.int64, nb.float32, nb.float64)
_IMAGE_SIGS = _loop_sigs((nb.uint32,))
_ARR_SIGS = _loop_sigs(_ARR_TYPES)
# `int32 + int32` is `int64` in numba.
_ADD_ARR_SIGS = _loop_sigs(_ARR_TYPES, (nb.int64, nb.int64, nb.float32, nb.float64))


# Elementwise loops over flat arrays, written out per operator. A cached
# `nb.guvectorize` would be shorter, but its disk cache segfaults when
# parallel processes fill a fresh cache.
@nb.jit(_IMAGE_SIGS, nogil=True, cache=True)
def _over_loop(src, dst, out):
    for i in range(out.size):
        out[i] = _over(src[i], dst[i])


@nb.jit(_IMAGE_SIGS, nogil=True, cache=True)
def _add_loop(src, dst, out):
    for i in range(out.size):
        out[i] = _add(src[i], dst[i])


@nb.jit(_IMAGE_SIGS, nogil=True, cache=True)
def _saturate_loop(src, dst, out):
    for i in range(out.size):
        out[i] = _saturate(src[i], dst[i])


@nb.jit(_IMAGE_SIGS, nogil=True, cache=True)
def _source_loop(src, dst, out):
    for i in range(out.size):
        out[i] = _source(src[i], dst[i])


@nb.jit(_ADD_ARR_SIGS, nogil=True, cache=True)
def _add_arr_loop(src, dst, out):
    for i in range(out.size):
        out[i] = _add_arr(src[i], dst[i])


@nb.jit(_ARR_SIGS, nogil=True, cache=True)
def _max_arr_loop(src, dst, out):
    for i in range(out.size):
        out[i] = _max_arr(src[i], dst[i])


@nb.jit(_ARR_SIGS, nogil=True, cache=True)
def _min_arr_loop(src, dst, out):
    for i in range(out.size):
        out[i] = _min_arr(src[i], dst[i])


@nb.jit(_ARR_SIGS, nogil=True, cache=True)
def _source_arr_loop(src, dst, out):
    for i in range(out.size):
        out[i] = _source_arr(src[i], dst[i])


def _apply(loop, src, dst, dtype, out_dtype):
    src, dst = np.asarray(src, dtype=dtype), np.asarray(dst, dtype=dtype)
    shape = np.broadcast_shapes(src.shape, dst.shape)
    src, dst = np.broadcast_to(src, shape), np.broadcast_to(dst, shape)
    out = np.empty(shape, dtype=out_dtype)
    loop(src.reshape(-1), dst.reshape(-1), out.reshape(-1))
    return out if out.ndim else out[()]


_ARR_DTYPES = tuple(map(np.dtype, ("int32", "int64", "float32", "float64")))


def _arr_dtype(src, dst):
    # The first supported type both inputs cast to safely, as ufunc loop selection does.
    dtype = np.result_type(src, dst)
    for t in _ARR_DTYPES:
        if np.can_cast(dtype, t):
            return t
    raise TypeError(f"Unsupported dtype for array composite operators: {dtype}")


def over(src, dst):
    return _apply(_over_loop, src, dst, np.uint32, np.uint32)


def add(src, dst):
    return _apply(_add_loop, src, dst, np.uint32, np.uint32)


def saturate(src, dst):
    return _apply(_saturate_loop, src, dst, np.uint32, np.uint32)


def source(src, dst):
    return _apply(_source_loop, src, dst, np.uint32, np.uint32)


def add_arr(src, dst):
    dtype = _arr_dtype(src, dst)
    # `int32 + int32` is `int64` in numba.
    out_dtype = np.dtype("int64") if dtype == np.int32 else dtype
    return _apply(_add_arr_loop, src, dst, dtype, out_dtype)


def max_arr(src, dst):
    dtype = _arr_dtype(src, dst)
    return _apply(_max_arr_loop, src, dst, dtype, dtype)


def min_arr(src, dst):
    dtype = _arr_dtype(src, dst)
    return _apply(_min_arr_loop, src, dst, dtype, dtype)


def source_arr(src, dst):
    dtype = _arr_dtype(src, dst)
    return _apply(_source_arr_loop, src, dst, dtype, dtype)

# Lookup table for storing compositing operators by function name
composite_op_lookup = dict(zip(
    image_operators + array_operators,
    (over, add, saturate, source, add_arr, max_arr, min_arr, source_arr),
))


@nb.jit(nogil=True, cache=True)
def _spread_image(arr, mask, out, code):
    """Spread kernel for images, compositing with ``image_operators[code]``.

    Module-level, so numba can cache it: a closure capturing the image
    operators gets a different cache key in every process.
    """
    M, N = arr.shape
    w = mask.shape[0]
    for y in range(M):
        for x in range(N):
            el = arr[y, x]
            # Skip if data is transparent
            if (int(el) >> 24) & 255:
                for i in range(w):
                    for j in range(w):
                        # Skip if mask is False at this value
                        if mask[i, j]:
                            if el == 0:
                                result = out[i + y, j + x]
                            if out[i + y, j + x] == 0:
                                result = el
                            else:
                                result = _image_op(code, el, out[i + y, j + x])
                            out[i + y, j + x] = result
