from __future__ import absolute_import, division, print_function

from itertools import count
from functools import reduce

from toolz import unique, concat, pluck, get, memoize
from dynd import nd
import numpy as np

from .aggregates import Summary
from .utils import ngjit, _exec


__all__ = ['compile_components']


@memoize
def compile_components(summary, schema):
    """
    Given a ``Summary`` object and a table schema, returns 5 sub-functions.

    Parameters
    ----------
    summary : Summary
        The expression describing the aggregations to be computed.

    Returns
    -------
    A tuple of the following functions:

    ``create(shape)``
        Takes the aggregate shape, and returns a tuple of initialized numpy
        arrays.

    ``info(df)``
        Takes a dataframe, and returns preprocessed 1D numpy arrays of the
        needed columns.

    ``append(i, x, y, *aggs_and_cols)``
        Appends the ``i``th row of the table to the ``(x, y)`` bin, given the
        base arrays and columns in ``aggs_and_cols``. This does the bulk of the
        work.

    ``combine(base_tuples)``
        Combine a list of base tuples into a single base tuple. This forms the
        reducing step in a reduction tree.

    ``finalize(aggs)``
        Given a tuple of base numpy arrays, returns the finalized
        ``dynd`` array.
    """
    paths, reds = zip(*preorder_traversal(summary))

    # List of base reductions (actually computed)
    bases = list(unique(concat(r._bases for r in reds)))
    dshapes = [b.out_dshape(schema) for b in bases]
    # List of tuples of (append, base, input columns, temps)
    calls = [_get_call_tuples(b, d) for (b, d) in zip(bases, dshapes)]
    # List of unique column names needed
    cols = list(unique(concat(pluck(2, calls))))
    # List of temps needed
    temps = list(pluck(3, calls))

    create = make_create(bases, dshapes)
    info = make_info(cols)
    append = make_append(bases, cols, calls)
    combine = make_combine(bases, dshapes, temps)
    finalize = make_finalize(bases, summary, schema)

    return create, info, append, combine, finalize


def preorder_traversal(summary):
    """Yields tuples of (path, reduction, dshape)"""
    for key, value in zip(summary.keys, summary.values):
        if isinstance(value, Summary):
            for key2, value2 in preorder_traversal(value):
                yield (key,) + key2, value2
        else:
            yield (key,), value


def _get_call_tuples(base, dshape):
    return base._build_append(dshape), (base,), base.inputs, base._temps


def make_create(bases, dshapes):
    creators = [b._build_create(d) for (b, d) in zip(bases, dshapes)]
    return lambda shape: tuple(c(shape) for c in creators)


def make_info(cols):
    return lambda df: tuple(c.apply(df) for c in cols)


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
    _exec(code, namespace)
    return ngjit(namespace['append'])


def make_combine(bases, dshapes, temps):
    arg_lk = dict((k, v) for (v, k) in enumerate(bases))
    calls = [(b._build_combine(d), [arg_lk[i] for i in (b,) + t])
             for (b, d, t) in zip(bases, dshapes, temps)]

    def combine(base_tuples):
        bases = tuple(np.stack(bs) for bs in zip(*base_tuples))
        return tuple(f(*get(inds, bases)) for (f, inds) in calls)

    return combine


def make_finalize(bases, summary, schema):
    dshape = str(summary.out_dshape(schema))
    paths = []
    finalizers = []
    indices = []
    arg_lk = dict((k, v) for (v, k) in enumerate(bases))
    for (path, red) in preorder_traversal(summary):
        paths.append(path)
        finalizers.append(red._build_finalize(schema))
        interms = red._bases
        indices.append([arg_lk[b] for b in interms])

    def finalize(bases):
        shape = bases[0].shape[:2]
        out = nd.empty(shape, dshape)
        for path, finalizer, inds in zip(paths, finalizers, indices):
            arr = reduce(getattr, path, out)
            if hasattr(arr.dtype, 'value_type'):
                arr = arr.view_scalars(arr.dtype.value_type)
            np_arr = nd.as_numpy(arr)
            np_arr[:] = finalizer(*get(inds, bases))
        return out

    return finalize
