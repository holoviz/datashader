from __future__ import annotations

import inspect
import warnings

import numba
import numpy as np
import pytest
from numba import jit

from datashader.macros import expand_varargs


# Example functions to test expand_varargs on
def function_no_vararg(a, b):
    return a + b


def function_with_vararg(a, b, *others):
    return a + b - function_no_vararg(*others)


def function_with_unsupported_vararg_use(a, b, *others):
    print(others[0])
    function_with_vararg(a, b, *others)


@jit(nopython=True, nogil=True)
def function_no_vararg_numba(a, b):
    return a + b


def function_with_vararg_call_numba(a, b, *others):
    return a + b - function_no_vararg_numba(*others)


def function_with_listcomp_vararg(z, *aggs_and_cols):
    """Function using list comprehension with starred vararg"""
    return process_items(*[ac[z] for ac in aggs_and_cols])

def process_items(*items):
    """Helper function that accepts items"""
    return sum(items)

def function_with_complex_listcomp(i, j, *items):
    """Function with more complex list comprehension"""
    return compute_result(i, j, *[item[i][j] for item in items])

def compute_result(i, j, *vals):
    """Helper function for complex case"""
    return i + j + sum(vals)


# Help functions
def get_args(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = inspect.getfullargspec(fn)

    args = spec.args
    if spec.varargs:
        args += ['*' + spec.varargs]

    return args


# Tests
def test_expand_varargs():
    assert get_args(function_with_vararg) == ['a', 'b', '*others']
    function_with_vararg_expanded = expand_varargs(2)(function_with_vararg)
    assert get_args(function_with_vararg_expanded) == ['a', 'b', '_0', '_1']

    assert (function_with_vararg(1, 2, 3, 4) ==
            function_with_vararg_expanded(1, 2, 3, 4))


def test_invalid_expand_number():
    with pytest.raises(ValueError) as e:
        # User forgets to construct decorator with expand_number
        expand_varargs(function_no_vararg)

    assert e.match(r"non\-negative integer")


def test_no_varargs_error():
    with pytest.raises(ValueError) as e:
        expand_varargs(2)(function_no_vararg)

    assert e.match(r"does not have a variable length positional argument")


def test_unsupported_vararg_use():
    with pytest.raises(ValueError) as e:
        expand_varargs(2)(function_with_unsupported_vararg_use)

    assert e.match(r"unsupported context")


def test_unsupported_expanding():
    # We could support this at some point, but this is to test we do not get
    # AttributeError: 'xxx' object has no attribute 'id'

    def func_with_bad_listcomp(*args):
        other_list = [1, 2, 3]
        return sum(*[x * 2 for x in other_list], args[0])

    with pytest.raises(ValueError, match="unsupported context"):
        expand_varargs(2)(func_with_bad_listcomp)


def test_listcomp_fails_without_expand_varargs():
    arr1 = [100, 101, 102]
    arr2 = [200, 201, 202]
    arr3 = [300, 301, 302]
    jit_fn = jit(nopython=True, nogil=True)(function_with_listcomp_vararg)
    with pytest.raises(numba.core.errors.TypingError):
        jit_fn(1, arr1, arr2, arr3)


def test_numba_jit_expanded_function():
    jit_fn = jit(nopython=True, nogil=True)(
        expand_varargs(2)(function_with_vararg_call_numba)
    )
    assert function_with_vararg_call_numba(1, 2, 3, 4) == jit_fn(1, 2, 3, 4)


def test_expand_varargs_with_listcomp():
    """Test expand_varargs with list comprehension pattern *[expr for var in vararg]"""
    # Create mock arrays (lists with __getitem__)
    arr1 = [100, 101, 102]
    arr2 = [200, 201, 202]
    arr3 = [300, 301, 302]

    # Test original function
    result_orig = function_with_listcomp_vararg(1, arr1, arr2, arr3)
    assert result_orig == 101 + 201 + 301  # sum of arr1[1], arr2[1], arr3[1]

    # Test expanded function
    function_expanded = expand_varargs(3)(function_with_listcomp_vararg)
    assert get_args(function_expanded) == ['z', '_0', '_1', '_2']

    result_expanded = function_expanded(1, arr1, arr2, arr3)
    assert result_orig == result_expanded


def test_expand_varargs_with_complex_listcomp():
    """Test expand_varargs with more complex list comprehension"""
    # Create 2D arrays
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    mat3 = [[9, 10], [11, 12]]

    # Test original function
    result_orig = function_with_complex_listcomp(1, 0, mat1, mat2, mat3)
    # i=1, j=0: mat1[1][0]=3, mat2[1][0]=7, mat3[1][0]=11
    # compute_result(1, 0, 3, 7, 11) = 1 + 0 + (3 + 7 + 11) = 22
    assert result_orig == 22

    # Test expanded function
    function_expanded = expand_varargs(3)(function_with_complex_listcomp)
    assert get_args(function_expanded) == ['i', 'j', '_0', '_1', '_2']

    result_expanded = function_expanded(1, 0, mat1, mat2, mat3)
    assert result_orig == result_expanded


def test_expand_varargs_listcomp_with_numba():
    """Test that expanded list comprehension works with numba"""
    @jit(nopython=True, nogil=True)
    def helper(*vals):
        total = 0
        for v in vals:
            total += v
        return total

    def func_with_listcomp(z, *arrays):
        return helper(*[arr[z] for arr in arrays])

    # Expand and jit the function
    jit_fn = jit(nopython=True, nogil=True)(
        expand_varargs(3)(func_with_listcomp)
    )

    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([10.0, 20.0, 30.0])
    arr3 = np.array([100.0, 200.0, 300.0])

    result = jit_fn(1, arr1, arr2, arr3)
    assert result == 222  # arr1[1] + arr2[1] + arr3[1]
