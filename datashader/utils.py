from __future__ import absolute_import, division, print_function

import sys
from inspect import getmro

from datashape import Unit, dshape
from datashape.predicates import launder
from datashape.typesets import real
import numba as nb
import numpy as np


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


class GeneratedFunction(nb.dispatcher.Dispatcher):
    def compile(self, sig):
        with self._compile_lock:
            args, return_type = nb.sigutils.normalize_signature(sig)
            existing = self.overloads.get(tuple(args))
            if existing is not None:
                return existing

            cres = self._cache.load_overload(sig, self.targetctx)
            if cres is not None:
                if not cres.objectmode and not cres.interpmode:
                    self.targetctx.insert_user_function(cres.entry_point,
                                                        cres.fndesc,
                                                        [cres.library])
                self.add_overload(cres)
                return cres.entry_point

            flags = nb.compiler.Flags()
            self.targetdescr.options.parse_as_flags(flags, self.targetoptions)

            cres = nb.compiler.compile_extra(self.typingctx, self.targetctx,
                                             self.py_func(*args), args=args,
                                             return_type=return_type,
                                             flags=flags, locals=self.locals)

            if cres.typing_error is not None and not flags.enable_pyobject:
                raise cres.typing_error

            self.add_overload(cres)
            self._cache.save_overload(sig, cres)
            return cres.entry_point


class GeneratedCPUDispatcher(GeneratedFunction):
    targetdescr = nb.targets.registry.CPUTarget()


nb.targets.registry.dispatcher_registry['gen_cpu'] = GeneratedCPUDispatcher


def generated_jit(*args, **kwargs):
    kwargs['target'] = 'gen_' + kwargs.get('target', 'cpu')
    return nb.jit(*args, **kwargs)


@generated_jit(nopython=True, nogil=True)
def is_missing(x):
    """Returns if the value is missing, per dynd's missing value flag
    semantics"""
    if isinstance(x, nb.types.Array):
        dt = x.dtype
    else:
        dt = x
    if isinstance(dt, nb.types.Integer):
        missing = np.iinfo(dt.name).min
        return lambda x: x == missing
    elif isinstance(dt, nb.types.Float):
        return lambda x: np.isnan(x)
    elif isinstance(x, nb.types.Array):
        return lambda x: np.full_like(x, False)
    else:
        return lambda x: False


def is_option(agg):
    """Returns if the dshape of the dynd array is an option type"""
    return hasattr(agg, 'value_type')


def dshape_from_dynd(ds):
    return dshape(str(ds))
