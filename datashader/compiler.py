from __future__ import division

from itertools import count
from functools import reduce

from toolz import unique, concat, pluck, juxt, get
from blaze.expr import Summary
from dynd import nd

from .reductions import (get_bases, get_create, get_cols, get_info, get_temps,
                         get_append, get_finalize)
from .util import ngjit, _exec
from .expr import optionify


__all__ = ['compile_reduction']


def compile_reduction(expr):
    """Given a `ByPixel` expression, returning 4 sub-functions.

    Parameters
    ----------
    expr : ByPixel
        The blaze expression describing the aggregations to be computed.

    Returns
    -------
    A tuple of the following functions:

    ``create(shape)``
        Takes the aggregate shape, and returns a tuple of initialized numpy arrays.

    ``info(df)``
        Takes a dataframe, and returns preprocessed 1D numpy arrays of the needed columns.

    ``append(i, x, y, *aggs_and_cols)``
        Appends the ``i``th row of the table to the ``(x, y)`` bin, given the
        base arrays and columns in ``aggs_and_cols``. This does the bulk of the
        work.

    ``finalize(aggs)``
        Given a tuple of base numpy arrays, return the finalized ``dynd`` array.
    """
    reductions = list(pluck(1, preorder_traversal(expr.apply)))
    bases = list(unique(concat(map(get_bases, reductions))))
    calls = list(map(_get_call_tuples, bases))
    cols = list(unique(concat(pluck(2, calls))))

    create = make_create(bases)
    info = make_info(cols)
    append = make_append(bases, cols, calls)
    finalize = make_finalize(bases, expr)

    return create, info, append, finalize


_get_call_tuples = juxt([get_append, lambda a: (a,), get_cols, get_temps])


def preorder_traversal(summary):
    """Yields tuples of (path, reduction)"""
    for name, value in zip(summary.names, summary.values):
        if isinstance(value, Summary):
            for name2, value2 in preorder_traversal(value):
                yield (name,) + name2, value2
        else:
            yield (name,), value


def make_create(bases):
    return juxt(map(get_create, bases))


def make_info(cols):
    return juxt(map(get_info, cols))


def make_append(bases, cols, calls):
    names = ('_{0}'.format(i) for i in count())
    inputs = list(bases) + list(cols)
    signature = [next(names) for i in inputs]
    arg_lk = dict(zip(inputs, signature))
    local_lk = {}
    namespace = {}
    body = []
    for func, bases, cols, temps in calls:
        local_lk.update(zip(temps, (next(names) for i in temps)))
        func_name = next(names)
        namespace[func_name] = func
        args = [arg_lk[i] for i in bases]
        args.extend('{0}[i]'.format(arg_lk[i]) for i in cols)
        args.extend([local_lk[i] for i in temps])
        body.append('{0}(x, y, {1})'.format(func_name, ', '.join(args)))
    body = ['{0} = {1}[y, x]'.format(name, arg_lk[agg])
            for agg, name in local_lk.items()] + body
    code = ('def append(i, x, y, {0}):\n'
            '    {1}').format(', '.join(signature), '\n    '.join(body))
    _exec(code, namespace, True)
    return ngjit(namespace['append'])


def make_finalize(bases, expr):
    dshape = str(expr.dshape.measure)
    paths = []
    finalizers = []
    indices = []
    arg_lk = dict((k, v) for (v, k) in enumerate(bases))
    for (path, red) in preorder_traversal(expr.apply):
        paths.append(path)
        finalizers.append(get_finalize(red))
        interms = get_bases(red)
        indices.append([arg_lk[b] for b in interms])

    def finalize(bases):
        shape = bases[0].shape[:2]
        out = nd.empty(shape, dshape)
        for path, finalizer, inds in zip(paths, finalizers, indices):
            reduce(getattr, path, out)[:] = finalizer(*get(inds, bases))
        return out

    return finalize
