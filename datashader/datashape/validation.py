"""
Datashape validation.
"""

from . import coretypes as T


def traverse(f, t):
    """
    Map `f` over `t`, calling `f` with type `t` and the map result of the
    mapping `f` over `t` 's parameters.

    Parameters
    ----------
    f : callable
    t : DataShape

    Returns
    -------
    DataShape
    """
    if isinstance(t, T.Mono) and not isinstance(t, T.Unit):
        return f(t, [traverse(f, p) for p in t.parameters])
    return t


def validate(ds):
    """
    Validate a datashape to see whether it is well-formed.

    Parameters
    ----------
    ds : DataShape

    Examples
    --------
    >>> from datashader.datashape import dshape
    >>> dshape('10 * int32')
    dshape("10 * int32")
    >>> dshape('... * int32')
    dshape("... * int32")
    >>> dshape('... * ... * int32') # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: Can only use a single wildcard
    >>> dshape('T * ... * X * ... * X') # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: Can only use a single wildcard
    >>> dshape('T * ...') # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    DataShapeSyntaxError: Expected a dtype
    """
    traverse(_validate, ds)


def _validate(ds, params):
    if isinstance(ds, T.DataShape):
        # Check ellipses
        ellipses = [x for x in ds.parameters if isinstance(x, T.Ellipsis)]
        if len(ellipses) > 1:
            raise TypeError("Can only use a single wildcard")
        elif isinstance(ds.parameters[-1], T.Ellipsis):
            raise TypeError("Measure may not be an Ellipsis (...)")
