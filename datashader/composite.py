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
            raise ValueError('Operator %r not one of the supported image operators: %s'
                            % (how, ', '.join(repr(el) for el in image_operators)))
    elif name not in array_operators:
        raise ValueError('Operator %r not one of the supported array operators: %s'
                        % (how, ', '.join(repr(el[:-4]) for el in array_operators)))


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

# Lookup table for storing compositing operators by function name
composite_op_lookup = {}


def operator(f):
    """Define and register a new image composite operator"""

    if jit_enabled:
        f2 = nb.vectorize(f)
        f2._compile_for_argtys((nb.types.uint32, nb.types.uint32))
        f2._frozen = True
    else:
        f2 = np.vectorize(f)

    composite_op_lookup[f.__name__] = f2
    return f2


@operator
def source(src, dst):
    if src & 0xff000000:
        return src
    else:
        return dst


@operator
def over(src, dst):
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


@operator
def add(src, dst):
    sr, sg, sb, sa = extract_scaled(src)
    dr, dg, db, da = extract_scaled(dst)

    a = min(1, sa + da)
    if a == 0:
        return np.uint32(0)
    r = (sr * sa + dr * da)/a
    g = (sg * sa + dg * da)/a
    b = (sb * sa + db * da)/a
    return combine_scaled(r, g, b, a)


@operator
def saturate(src, dst):
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



def arr_operator(f):
    """Define and register a new array composite operator"""

    if jit_enabled:
        f2 = nb.vectorize(f)
        f2._compile_for_argtys(
           (nb.types.int32, nb.types.int32))
        f2._compile_for_argtys(
           (nb.types.int64, nb.types.int64))
        f2._compile_for_argtys(
            (nb.types.float32, nb.types.float32))
        f2._compile_for_argtys(
            (nb.types.float64, nb.types.float64))
        f2._frozen = True
    else:
        f2 = np.vectorize(f)

    composite_op_lookup[f.__name__] = f2
    return f2


@arr_operator
def source_arr(src, dst):
    if src:
        return src
    else:
        return dst

@arr_operator
def add_arr(src, dst):
    return src + dst

@arr_operator
def max_arr(src, dst):
    return max([src,  dst])

@arr_operator
def min_arr(src, dst):
    return min([src,  dst])
