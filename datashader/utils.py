from __future__ import absolute_import, division, print_function

import sys
from inspect import getmro
from contextlib import contextmanager

from datashape import Unit
from datashape.predicates import launder
from datashape.typesets import real
import numba as nb


ngjit = nb.jit(nopython=True, nogil=True)


if sys.version_info.major == 3:
    def __exec(codestr, glbls):
        exec(codestr, glbls)
else:
    eval(compile("""
def __exec(codestr, glbls):
    exec codestr in glbls
""",
                 "<_exec>", "exec"))


def _exec(code, namespace, debug=False):
    if debug:
        print(("Code:\n-----\n{0}\n"
               "Namespace:\n----------\n{1}").format(code, namespace))
    __exec(code, namespace)


class Dispatcher(object):
    """Simple single dispatch."""
    def __init__(self):
        self._lookup = {}

    def register(self, type, func=None):
        """Register dispatch of `func` on arguments of type `type`"""
        if func is None:
            return lambda f: self.register(type, f)
        if isinstance(type, tuple):
            for t in type:
                self.register(t, func)
        else:
            self._lookup[type] = func
        return func

    def __call__(self, head, *rest, **kwargs):
        # We dispatch first on type(head), and fall back to iterating through
        # the mro. This is significantly faster in the common case where
        # type(head) is in the lookup, with only a small penalty on fall back.
        lk = self._lookup
        typ = type(head)
        if typ in lk:
            return lk[typ](head, *rest, **kwargs)
        for cls in getmro(typ)[1:]:
            if cls in lk:
                return lk[cls](head, *rest, **kwargs)
        raise TypeError("No dispatch for {0} type".format(typ))


def isreal(dt):
    dt = launder(dt)
    return isinstance(dt, Unit) and dt in real
