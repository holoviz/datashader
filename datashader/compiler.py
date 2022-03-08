from __future__ import absolute_import, division, print_function

from itertools import count

from toolz import unique, concat, pluck, get, memoize
import numpy as np
import xarray as xr

from .compatibility import _exec
from .reductions import by, category_codes, summary
from .utils import ngjit


__all__ = ['compile_components']


@memoize
def compile_components(agg, schema, glyph, cuda=False):
    """Given a ``Aggregation`` object and a schema, return 5 sub-functions.

    Parameters
    ----------
    agg : Aggregation
        The expression describing the aggregation(s) to be computed.

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
        Given a tuple of base numpy arrays, returns the finalized ``DataArray``
        or ``Dataset``.
    """
    reds = list(traverse_aggregation(agg))

    # List of base reductions (actually computed)
    bases = list(unique(concat(r._build_bases(cuda) for r in reds)))
    dshapes = [b.out_dshape(schema) for b in bases]
    # List of tuples of (append, base, input columns, temps)
    calls = [_get_call_tuples(b, d, schema, cuda) for (b, d) in zip(bases, dshapes)]
    # List of unique column names needed
    cols = list(unique(concat(pluck(2, calls))))
    # List of temps needed
    temps = list(pluck(3, calls))

    create = make_create(bases, dshapes, cuda)
    info = make_info(cols)
    append = make_append(bases, cols, calls, glyph, isinstance(agg, by))
    combine = make_combine(bases, dshapes, temps)
    finalize = make_finalize(bases, agg, schema, cuda)

    return create, info, append, combine, finalize


def traverse_aggregation(agg):
    """Yield a left->right traversal of an aggregation"""
    if isinstance(agg, summary):
        for a in agg.values:
            for a2 in traverse_aggregation(a):
                yield a2
    else:
        yield agg


def _get_call_tuples(base, dshape, schema, cuda):
    return (base._build_append(dshape, schema, cuda),
            (base,), base.inputs, base._build_temps(cuda))


def make_create(bases, dshapes, cuda):
    creators = [b._build_create(d) for (b, d) in zip(bases, dshapes)]
    if cuda:
        import cupy
        array_module = cupy
    else:
        array_module = np
    return lambda shape: tuple(c(shape, array_module) for c in creators)


def make_info(cols):
    return lambda df: tuple(c.apply(df) for c in cols)


def make_append(bases, cols, calls, glyph, categorical):
    names = ('_{0}'.format(i) for i in count())
    inputs = list(bases) + list(cols)
    signature = [next(names) for i in inputs]
    arg_lk = dict(zip(inputs, signature))
    local_lk = {}
    namespace = {}
    body = []
    ndims = glyph.ndims
    if ndims is not None:
        subscript = ', '.join(['i' + str(n) for n in range(ndims)])
    else:
        subscript = None

    for func, bases, cols, temps in calls:
        local_lk.update(zip(temps, (next(names) for i in temps)))
        func_name = next(names)
        namespace[func_name] = func
        args = [arg_lk[i] for i in bases]
        if categorical and isinstance(cols[0], category_codes):
            pass
        elif ndims is None:
            args.extend('{0}'.format(arg_lk[i]) for i in cols)
        elif categorical:
            args.extend('{0}[{1}][1]'.format(arg_lk[i], subscript)
                        for i in cols)
        else:
            args.extend('{0}[{1}]'.format(arg_lk[i], subscript)
                        for i in cols)

        args.extend([local_lk[i] for i in temps])
        body.append('{0}(x, y, {1})'.format(func_name, ', '.join(args)))

    body = ['{0} = {1}[y, x]'.format(name, arg_lk[agg])
            for agg, name in local_lk.items()] + body

    # Categorical aggregate arrays need to be unpacked
    if categorical:
        col_index = '' if isinstance(cols[0], category_codes) else '[0]'
        cat_var = 'cat = int({0}[{1}]{2})'.format(signature[-1], subscript, col_index)
        aggs = ['{0} = {0}[:, :, cat]'.format(s) for s in signature[:len(calls)]]
        body = [cat_var] + aggs + body

    if ndims is None:
        code = ('def append(x, y, {0}):\n'
                '    {1}').format(', '.join(signature), '\n    '.join(body))
    else:
        code = ('def append({0}, x, y, {1}):\n'
                '    {2}'
                ).format(subscript, ', '.join(signature), '\n    '.join(body))
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


def make_finalize(bases, agg, schema, cuda):
    arg_lk = dict((k, v) for (v, k) in enumerate(bases))
    if isinstance(agg, summary):
        calls = []
        for key, val in zip(agg.keys, agg.values):
            f = make_finalize(bases, val, schema, cuda)
            try:
                # Override bases if possible
                bases = val._build_bases(cuda)
            except AttributeError:
                pass
            inds = [arg_lk[b] for b in bases]
            calls.append((key, f, inds))

        def finalize(bases, cuda=False, **kwargs):
            data = {key: finalizer(get(inds, bases), cuda, **kwargs)
                    for (key, finalizer, inds) in calls}
            return xr.Dataset(data)
        return finalize
    else:
        return agg._build_finalize(schema)
