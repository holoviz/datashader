from __future__ import annotations
import warnings
import pytest

from datashader.macros import expand_varargs
import inspect
from numba import jit


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


def test_numba_jit_expanded_function():
    jit_fn = jit(nopython=True, nogil=True)(
        expand_varargs(2)(function_with_vararg_call_numba)
    )
    assert function_with_vararg_call_numba(1, 2, 3, 4) == jit_fn(1, 2, 3, 4)
