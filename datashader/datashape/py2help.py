import sys
import itertools

# Portions of this taken from the six library, licensed as follows.
#
# Copyright (c) 2010-2013 Benjamin Peterson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import platform

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

CPYTHON = platform.python_implementation() == 'CPython'

if PY2:
    import __builtin__
    reduce = __builtin__.reduce
    _inttypes = (int, long)
    unicode = __builtin__.unicode
    basestring = __builtin__.basestring
    _strtypes = (str, unicode)

    from types import DictProxyType as MappingProxyType

    if CPYTHON:
        from ctypes import pythonapi, py_object

        mappingproxy = pythonapi.PyDictProxy_New
        mappingproxy.argtypes = [py_object]
        mappingproxy.restype = py_object
        del pythonapi
        del py_object
    else:
        # TODO: Figure out how to make these on pypy.
        # If this gets done, please update the skipif condition in:
        # test_discovery:test_mappingproxy
        def mappingproxy(ob):
            raise ValueError('cannot create mapping proxies in py2 on pypy')

else:
    from functools import reduce
    _inttypes = (int,)
    unicode = str
    basestring = str
    _strtypes = (str,)

    from types import MappingProxyType
    mappingproxy = MappingProxyType


def with_metaclass(metaclass, *bases):
    """Helper for using metaclasses in a py2/3 compatible way.

    Parameters
    ----------
    metaclass : type
        The metaclass to apply.
    bases : iterable of type
        The types to subclass.

    Notes
    -----
    The translations for python 2 and 3 look like:

    ::
        # Compat
        class C(with_metaclass(M, A, B)):
            pass

        # Pyton 2
        class C(A, B):
            __metaclass__ = M

        # Python 3
        class C(A, B, metaclass=M):
            pass
    """
    return metaclass('_', bases, {})


try:
    from collections import OrderedDict
except ImportError:
    class OrderedDict(object):
        def __new__(cls, *args, **kwargs):
            raise TypeError('OrderedDict not supported before python 2.7')
