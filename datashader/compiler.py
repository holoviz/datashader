from __future__ import annotations

from itertools import count

from toolz import unique, concat, pluck, get, memoize
import numpy as np
import xarray as xr

from .reductions import by, category_codes, summary, where
from .utils import ngjit


__all__ = ['compile_components']


@memoize
def compile_components(agg, schema, glyph, *, antialias=False, cuda=False, partitioned=False):
    """Given an ``Aggregation`` object and a schema, return 5 sub-functions
    and information on how to perform the second stage aggregation if
    antialiasing is requested,

    Parameters
    ----------
    agg : Aggregation
        The expression describing the aggregation(s) to be computed.

    schema : DataShape
        Columns and dtypes in the source dataset.

    glyph : Glyph
        The glyph to render.

    antialias : bool
        Whether to render using antialiasing.

    cuda : bool
        Whether to render using CUDA (on the GPU) or CPU.

    partitioned : bool
        Whether the source dataset is partitioned using dask.

    Returns
    -------
    A tuple of the following functions:

    ``create(shape)``
        Takes the aggregate shape, and returns a tuple of initialized numpy
        arrays.

    ``info(df, canvas_shape)``
        Takes a dataframe, and returns preprocessed 1D numpy arrays of the
        needed columns.

    ``append(i, x, y, *aggs_and_cols)``
        Appends the ``i``th row of the table to the ``(x, y)`` bin, given the
        base arrays and columns in ``aggs_and_cols``. This does the bulk of the
        work.

    ``combine(base_tuples)``
        Combine a list of base tuples into a single base tuple. This forms the
        reducing step in a reduction tree.

    ``finalize(aggs, cuda)``
        Given a tuple of base numpy arrays, returns the finalized ``DataArray``
        or ``Dataset``.

    ``antialias_stage_2``
        If using antialiased lines this is a tuple of the ``AntialiasCombination``
        values corresponding to the aggs. If not using antialiased lines then
        this is False.
    """
    reds = list(traverse_aggregation(agg))

    # List of base reductions (actually computed)
    bases = list(unique(concat(r._build_bases(cuda, partitioned) for r in reds)))
    dshapes = [b.out_dshape(schema, antialias, cuda, partitioned) for b in bases]

    # Information on how to perform second stage aggregation of antialiased lines,
    # including whether antialiased lines self-intersect or not as we need a single
    # value for this even for a compound reduction. This is by default True, but
    # is False if a single constituent reduction requests it.
    if antialias:
        self_intersect, antialias_stage_2 = make_antialias_stage_2(reds, bases)
        if cuda:
            import cupy
            array_module = cupy
        else:
            array_module = np
        antialias_stage_2 = antialias_stage_2(array_module)
    else:
        self_intersect = False
        antialias_stage_2 = False

    # List of tuples of (append, base, input columns, temps, combine temps, uses cuda mutex)
    calls = [_get_call_tuples(b, d, schema, cuda, antialias, self_intersect, partitioned)
             for (b, d) in zip(bases, dshapes)]

    # List of unique column names needed
    cols = list(unique(concat(pluck(2, calls))))
    # List of temps needed
    temps = list(pluck(3, calls))
    combine_temps = list(pluck(4, calls))

    create = make_create(bases, dshapes, cuda)
    append, uses_cuda_mutex = make_append(bases, cols, calls, glyph, isinstance(agg, by), antialias)
    info = make_info(cols, uses_cuda_mutex)
    combine = make_combine(bases, dshapes, temps, combine_temps, antialias, cuda, partitioned)
    finalize = make_finalize(bases, agg, schema, cuda, partitioned)

    return create, info, append, combine, finalize, antialias_stage_2


def traverse_aggregation(agg):
    """Yield a left->right traversal of an aggregation"""
    if isinstance(agg, summary):
        for a in agg.values:
            for a2 in traverse_aggregation(a):
                yield a2
    else:
        yield agg


def _get_call_tuples(base, dshape, schema, cuda, antialias, self_intersect, partitioned):
    # Comments refer to usage in make_append()
    return (
        base._build_append(dshape, schema, cuda, antialias, self_intersect),  # func
        (base,),  # bases
        base.inputs,  # cols
        base._build_temps(cuda),  # temps
        base._build_combine_temps(cuda, partitioned),  # combine temps
        cuda and base.uses_cuda_mutex(),  # uses cuda mutex
    )


def make_create(bases, dshapes, cuda):
    creators = [b._build_create(d) for (b, d) in zip(bases, dshapes)]
    if cuda:
        import cupy
        array_module = cupy
    else:
        array_module = np
    return lambda shape: tuple(c(shape, array_module) for c in creators)


def make_info(cols, uses_cuda_mutex):
    def info(df, canvas_shape):
        ret = tuple(c.apply(df) for c in cols)
        if uses_cuda_mutex:
            import cupy  # Guaranteed to be available if uses_cuda_mutex is True
            import numba
            from packaging.version import Version
            if Version(numba.__version__) >= Version("0.57"):
                mutex_array = cupy.zeros(canvas_shape, dtype=np.uint32)
            else:
                mutex_array = cupy.zeros((1,), dtype=np.uint32)
            ret += (mutex_array,)
        return ret

    return info


def make_append(bases, cols, calls, glyph, categorical, antialias):
    names = ('_{0}'.format(i) for i in count())
    inputs = list(bases) + list(cols)
    any_uses_cuda_mutex = any(call[5] for call in calls)
    if any_uses_cuda_mutex:
        # This adds an argument to the append() function that is the cuda mutex
        # generated in make_info.
        inputs += ["_cuda_mutex"]
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

    for func, bases, cols, temps, _, uses_cuda_mutex in calls:
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
        if antialias:
            args.append("aa_factor")

        if uses_cuda_mutex:
            args.append(arg_lk["_cuda_mutex"])

        where_reduction = len(bases) == 1 and isinstance(bases[0], where)
        if where_reduction:
            update_index_arg_name = next(names)
            args.append(update_index_arg_name)

            # where reduction needs access to the return of the contained
            # reduction, which is the preceding one here.
            body[-1] = f'{update_index_arg_name} = {body[-1]}'

            # If the lookup_column is None then it is a row index, and all row
            # indexes passed to append() are valid, i.e. >= 0.
            # If the lookup_column is a real column then we need to check if
            # the value passed to append() is NaN, and if so do nothing.
            lookup_column = bases[0].column
            if lookup_column is None:
                whitespace = ''
            else:
                var = args[1]
                prev_body = body[-1]
                body[-1] = f'if {var}<=0 or {var}>0:'  # Inline CUDA-friendly 'is not nan' test
                body.append(f'    {prev_body}')
                whitespace = '    '

            body.append(f'{whitespace}if {update_index_arg_name} >= 0:')
            call  = f'    {whitespace}{func_name}(x, y, {", ".join(args)})'
        else:
            call  = f'{func_name}(x, y, {", ".join(args)})'

        body.append(call)

    body = ['{0} = {1}[y, x]'.format(name, arg_lk[agg])
            for agg, name in local_lk.items()] + body

    # Categorical aggregate arrays need to be unpacked
    if categorical:
        col_index = '' if isinstance(cols[0], category_codes) else '[0]'
        cat_var = 'cat = int({0}[{1}]{2})'.format(signature[-1], subscript, col_index)
        aggs = ['{0} = {0}[:, :, cat]'.format(s) for s in signature[:len(calls)]]
        body = [cat_var] + aggs + body

    if antialias:
        signature.insert(0, "aa_factor")

    if ndims is None:
        code = ('def append(x, y, {0}):\n'
                '    {1}').format(', '.join(signature), '\n    '.join(body))
    else:
        code = ('def append({0}, x, y, {1}):\n'
                '    {2}'
                ).format(subscript, ', '.join(signature), '\n    '.join(body))
    exec(code, namespace)
    return ngjit(namespace['append']), any_uses_cuda_mutex


def make_combine(bases, dshapes, temps, combine_temps, antialias, cuda, partitioned):
    arg_lk = dict((k, v) for (v, k) in enumerate(bases))

    # where._combine() deals with combine of preceding reduction so exclude
    # it from explicit combine calls.
    base_is_where = [isinstance(b, where) for b in bases]
    next_base_is_where = base_is_where[1:] + [False]
    calls = [(None if n else b._build_combine(d, antialias, cuda, partitioned), [arg_lk[i] for i in (b,) + t + ct])
             for (b, d, t, ct, n) in zip(bases, dshapes, temps, combine_temps, next_base_is_where)]

    def combine(base_tuples):
        bases = tuple(np.stack(bs) for bs in zip(*base_tuples))
        ret = []
        for is_where, (func, inds) in zip(base_is_where, calls):
            if func is None:
                continue
            call = func(*get(inds, bases))
            if is_where:
                # Separate aggs of where reduction and its selector,
                # selector's goes first to match order of bases.
                ret.extend(call[::-1])
            else:
                ret.append(call)
        return tuple(ret)

    return combine


def make_finalize(bases, agg, schema, cuda, partitioned):
    arg_lk = dict((k, v) for (v, k) in enumerate(bases))
    if isinstance(agg, summary):
        calls = []
        for key, val in zip(agg.keys, agg.values):
            f = make_finalize(bases, val, schema, cuda, partitioned)
            try:
                # Override bases if possible
                bases = val._build_bases(cuda, partitioned)
            except AttributeError:
                pass
            inds = [arg_lk[b] for b in bases]
            calls.append((key, f, inds))

        def finalize(bases, cuda=False, **kwargs):
            data = {key: finalizer(get(inds, bases), cuda, **kwargs)
                    for (key, finalizer, inds) in calls}

            # Copy x and y range attrs from any DataArray (their ranges are all the same)
            # to set on parent Dataset
            name = agg.keys[0]  # Name of first DataArray.
            attrs = {attr: data[name].attrs[attr] for attr in ('x_range', 'y_range')}

            return xr.Dataset(data, attrs=attrs)
        return finalize
    else:
        return agg._build_finalize(schema)


def make_antialias_stage_2(reds, bases):
    # Only called if antialias is True.

    # Prefer a single-stage antialiased aggregation, but if any requested
    # reduction requires two stages then force use of two for all reductions.
    self_intersect = True
    for red in reds:
        if red._antialias_requires_2_stages():
            self_intersect = False
            break

    def antialias_stage_2(array_module):
        return tuple(zip(*concat(b._antialias_stage_2(self_intersect, array_module) for b in bases)))

    return self_intersect, antialias_stage_2
