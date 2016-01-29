from __future__ import absolute_import, division, print_function

import sys
import re
from inspect import getmro
from keyword import iskeyword

import numba as nb
import numpy as np
from datashape import Unit, dshape
from datashape.predicates import launder
from datashape.typesets import real
from dynd import nd


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


# Dynd Missing Type Flags
dynd_missing_types = {np.dtype('bool'): 2,
                      np.dtype('i2'): np.iinfo('i2').min,
                      np.dtype('i4'): np.iinfo('i4').min,
                      np.dtype('i8'): np.iinfo('i8').min,
                      np.dtype('f4'): np.nan,
                      np.dtype('f8'): np.nan}


def dynd_to_np_mask(x):
    if is_option(x.dtype):
        arr = nd.as_numpy(x.view_scalars(x.dtype.value_type))
        missing = is_missing(arr)
    else:
        arr = nd.as_numpy(x)
        missing = np.full_like(arr, False, dtype='bool')
    return arr, missing


def is_option(agg):
    """Returns if the dshape of the dynd array is an option type"""
    return hasattr(agg, 'value_type')


def dshape_from_dynd(ds):
    return dshape(str(ds))


def is_valid_identifier(s):
    """Check whether a string is a valid Python identifier.

    Examples
    --------
    >>> is_valid_identifier('foo')
    True
    >>> is_valid_identifier('foo bar')
    False
    >>> is_valid_identifier('1foo')
    False
    >>> is_valid_identifier('foo1')
    True
    >>> is_valid_identifier('for')
    False
    """
    return (not iskeyword(s) and
            re.match(r'^[_a-zA-Z][_a-zA-Z0-9]*$', s) is not None)
