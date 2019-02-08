from __future__ import absolute_import
from datashader.utils import ngjit
from numba import vectorize, int64
import numpy as np
import os

"""
Initially based on https://github.com/galtay/hilbert_curve, but specialized
for 2 dimensions with numba acceleration
"""

NUMBA_DISABLE_JIT = os.environ.get('NUMBA_DISABLE_JIT', 0)


@ngjit
def _int_2_binary(n, width):
    """Return a binary byte array representation of `n` zero padded to `width`
    bits."""
    res = np.zeros(width, dtype=np.uint8)
    i = 0
    for i in range(width):
        res[width - i - 1] = n % 2
        n = n >> 1
    return res


@ngjit
def _binary_2_int(bin_vec):
    """Convert a binary byte array to an integer"""
    res = 0
    next_val = 1
    width = len(bin_vec)
    for i in range(width):
        res += next_val*bin_vec[width - i - 1]
        next_val <<= 1
    return res


@ngjit
def _hilbert_integer_to_transpose(p, h):
    """Store a hilbert integer (`h`) as its transpose (`x`).

    Args:
        p (int): iterations to use in the hilbert curve
        h (int): integer distance along hilbert curve
    Returns:
        x (list): transpose of h
                  (n components with values between 0 and 2**p-1)
    """
    n = 2
    h_bits = _int_2_binary(h, p * n)

    x = [_binary_2_int(h_bits[i::n]) for i in range(n)]
    return x


@ngjit
def _transpose_to_hilbert_integer(p, x, y):
    """Restore a hilbert integer (`h`) from its transpose (`x`).

    Args:
        p (int): iterations to use in the hilbert curve
        x (list): transpose of h
                  (n components with values between 0 and 2**p-1)

    Returns:
        h (int): integer distance along hilbert curve
    """
    bin1 = _int_2_binary(x, p)
    bin2 = _int_2_binary(y, p)
    concat = np.zeros(2*p, dtype=np.uint8)
    for i in range(p):
        concat[2*i] = bin1[i]
        concat[2*i+1] = bin2[i]

    h = _binary_2_int(concat)
    return h


@ngjit
def coordinates_from_distance(p, h):
    """Return the coordinates for a given hilbert distance.

    Args:
        p (int): iterations to use in the hilbert curve
        h (int): integer distance along hilbert curve
    Returns:
        x (list): transpose of h
                  (n components with values between 0 and 2**p-1)
    """

    n = 2
    x = _hilbert_integer_to_transpose(p, h)
    Z = 2 << (p-1)

    # Gray decode by H ^ (H/2)
    t = x[n-1] >> 1
    for i in range(n-1, 0, -1):
        x[i] ^= x[i-1]
    x[0] ^= t

    # Undo excess work
    Q = 2
    while Q != Z:
        P = Q - 1
        for i in range(n-1, -1, -1):
            if x[i] & Q:
                # invert
                x[0] ^= P
            else:
                # exchange
                t = (x[0] ^ x[i]) & P
                x[0] ^= t
                x[i] ^= t
        Q <<= 1

    # done
    return x


if NUMBA_DISABLE_JIT:
    vect = np.vectorize
else:
    vect = vectorize([int64(int64, int64, int64)], nopython=True)


@vect
def distance_from_coordinates(p, x, y):
    """Return the hilbert distance for a given set of coordinates.

    Args:
        p (int): iterations to use in the hilbert curve
        x_in (list): transpose of h
                     (n components with values between 0 and 2**p-1)

    Returns:
        h (int): integer distance along hilbert curve
    """
    n = 2

    x = np.array([x, y], dtype=np.int64)

    M = 1 << (p - 1)

    # Inverse undo excess work
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            if x[i] & Q:
                x[0] ^= P
            else:
                t = (x[0] ^ x[i]) & P
                x[0] ^= t
                x[i] ^= t
        Q >>= 1

    # Gray encode
    for i in range(1, n):
        x[i] ^= x[i-1]
    t = 0
    Q = M
    while Q > 1:
        if x[n-1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(n):
        x[i] ^= t

    h = _transpose_to_hilbert_integer(p, x[0], x[1])
    return h
