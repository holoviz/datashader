from __future__ import division

import numba as nb
import numpy as np


__all__ = ('composite_op_lookup', 'over', 'add', 'saturate', 'source')


@nb.jit('(uint32,)', nopython=True, nogil=True, cache=True)
def extract_scaled(x):
    """Extract components as float64 values in [0.0, 1.0]"""
    r = np.float64((x & 255) / 255)
    g = np.float64(((x >> 8) & 255) / 255)
    b = np.float64(((x >> 16) & 255) / 255)
    a = np.float64(((x >> 24) & 255) / 255)
    return r, g, b, a


@nb.jit('(float64, float64, float64, float64)', nopython=True,
        nogil=True, cache=True)
def combine_scaled(r, g, b, a):
    """Combine components in [0, 1] to rgba uint32"""
    r2 = np.uint32(r * 255)
    g2 = np.uint32(g * 255)
    b2 = np.uint32(b * 255)
    a2 = np.uint32(a * 255)
    return np.uint32((a2 << 24) | (b2 << 16) | (g2 << 8) | r2)


extract_scaled.disable_compile()
combine_scaled.disable_compile()


# Lookup table for storing compositing operators by function name
composite_op_lookup = {}


def operator(f):
    """Define and register a new composite operator"""
    f2 = nb.vectorize(f)
    f2._compile_for_argtys((nb.types.uint32, nb.types.uint32))
    f2._frozen = True
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
