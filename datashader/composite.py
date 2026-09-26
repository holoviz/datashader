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
_image_scalars = (_over, _add, _saturate, _source)
_arr_scalars = (_add_arr, _max_arr, _min_arr, _source_arr)


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


# Numba caches a gufunc's loop wrapper on disk, but not a `nb.vectorize` one.
# Explicit signatures, so a warm import doesn't load the scalar kernels.
def _image_ufunc(code):
    if not jit_enabled:
        return np.vectorize(_image_scalars[code])

    def kernel(src, dst, out):
        out[0] = _image_op(code, src, dst)

    kernel.__name__ = image_operators[code]
    return nb.guvectorize(["void(uint32, uint32, uint32[:])"], "(),()->()", cache=True)(kernel)


_ARR_TYPES = ("int32", "int64", "float32", "float64")


def _arr_ufunc(code, out_types=_ARR_TYPES):
    if not jit_enabled:
        return np.vectorize(_arr_scalars[code])

    def kernel(src, dst, out):
        out[0] = _arr_op(code, src, dst)

    kernel.__name__ = array_operators[code]
    sigs = [f"void({t}, {t}, {o}[:])" for t, o in zip(_ARR_TYPES, out_types)]
    return nb.guvectorize(sigs, "(),()->()", cache=True)(kernel)


over, add, saturate, source = map(_image_ufunc, range(len(image_operators)))
# `int32 + int32` is `int64` in numba.
add_arr = _arr_ufunc(0, out_types=("int64", "int64", "float32", "float64"))
max_arr, min_arr, source_arr = map(_arr_ufunc, range(1, len(array_operators)))

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
