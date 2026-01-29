from abc import ABC

from ..coretypes import (
    DataShape,
    DateTime,
    Function,
    Option,
    Record,
    String,
    Time,
    TimeDelta,
    Tuple,
    Units,
)
from ..dispatch import dispatch


def _fmt_path(path):
    """Format the path for final display.

    Parameters
    ----------
    path : iterable of str
        The path to the values that are not equal.

    Returns
    -------
    fmtd : str
        The formatted path to put into the error message.
    """
    if not path:
        return ''
    return 'path: _' + ''.join(path)


@dispatch(DataShape, DataShape)
def assert_dshape_equal(a, b, check_dim=True, path=None, **kwargs):
    """Assert that two dshapes are equal, providing an informative error
    message when they are not equal.

    Parameters
    ----------
    a, b : dshape
        The dshapes to check for equality.
    check_dim : bool, optional
        Check shapes for equality with respect to their dimensions.
        default: True
    check_tz : bool, optional
        Checks times and datetimes for equality with respect to timezones.
        default: True
    check_timedelta_unit : bool, optional
        Checks timedeltas for equality with respect to their unit (us, ns, ...).
        default: True
    check_str_encoding : bool, optional
        Checks strings for equality with respect to their encoding.
        default: True
    check_str_fixlen : bool, optional
        Checks string for equality with respect to their fixlen.
        default: True
    check_record_order : bool, optional
        Checks records for equality with respect to the order of the fields.
        default: True

    Raises
    ------
    AssertionError
        Raised when the two dshapes are not equal.
    """
    ashape = a.shape
    bshape = b.shape

    if path is None:
        path = ()

    if check_dim:
        for n, (adim, bdim) in enumerate(zip(ashape, bshape)):
            if adim != bdim:
                path += f'.shape[{n}]',
                raise AssertionError(
                    'dimensions do not match: {} != {}{}\n{}'.format(
                        adim,
                        bdim,
                        ('\n{} != {}'.format(
                            ' * '.join(map(str, ashape)),
                            ' * '.join(map(str, bshape)),
                        )) if len(a.shape) > 1 else '',
                        _fmt_path(path),
                    ),
                )

    path += '.measure',
    assert_dshape_equal(
        a.measure,
        b.measure,
        check_dim=check_dim,
        path=path,
        **kwargs
    )


class Slotted(ABC):
    @classmethod
    def __subclasshook__(cls, subcls):
        return hasattr(subcls, '__slots__')


@assert_dshape_equal.register(Slotted, Slotted)
def _check_slots(a, b, path=None, **kwargs):
    if type(a) is not type(b):
        return _base_case(a, b, path=path, **kwargs)

    msg = f'slots mismatch: {a.__slots__!r} != {b.__slots__!r}\n{_fmt_path(path)}'
    assert a.__slots__ == b.__slots__, msg
    if path is None:
        path = ()
    for slot in a.__slots__:
        assert getattr(a, slot) == getattr(b, slot), \
            "{} {}s do not match: {!r} != {!r}\n{}".format(
                type(a).__name__.lower(),
                slot,
                getattr(a, slot),
                getattr(b, slot),
                _fmt_path(path + ('.' + slot,)),
            )


@assert_dshape_equal.register(object, object)
def _base_case(a, b, path=None, **kwargs):
    assert a == b, f'{a} != {b}\n{_fmt_path(path)}'


@dispatch((DateTime, Time), (DateTime, Time))
def assert_dshape_equal(a, b, path=None, check_tz=True, **kwargs):
    if type(a) is not type(b):
        return _base_case(a, b)
    if check_tz:
        _check_slots(a, b, path)


@dispatch(TimeDelta, TimeDelta)
def assert_dshape_equal(a, b, path=None, check_timedelta_unit=True, **kwargs):
    if check_timedelta_unit:
        _check_slots(a, b, path)


@dispatch(Units, Units)
def assert_dshape_equal(a, b, path=None, **kwargs):
    if path is None:
        path = ()

    assert a.unit == b.unit, '{} units do not match: {!r} != {}\n{}'.format(
        type(a).__name__.lower(), a.unit, b.unit, _fmt_path(path + ('.unit',)),
    )

    path.append('.tp')
    assert_dshape_equal(a.tp, b.tp, **kwargs)


@dispatch(String, String)
def assert_dshape_equal(a,
                        b,
                        path=None,
                        check_str_encoding=True,
                        check_str_fixlen=True,
                        **kwargs):
    if path is None:
        path = ()
    if check_str_encoding:
        assert a.encoding == b.encoding, \
            'string encodings do not match: {!r} != {!r}\n{}'.format(
                a.encoding, b.encoding, _fmt_path(path + ('.encoding',)),
            )

    if check_str_fixlen:
        assert a.fixlen == b.fixlen, \
            'string fixlens do not match: {} != {}\n{}'.format(
                a.fixlen, b.fixlen, _fmt_path(path + ('.fixlen',)),
            )


@dispatch(Option, Option)
def assert_dshape_equal(a, b, path=None, **kwargs):
    if path is None:
        path = ()
    path += '.ty',
    return assert_dshape_equal(a.ty, b.ty, path=path, **kwargs)


@dispatch(Record, Record)
def assert_dshape_equal(a, b, check_record_order=True, path=None, **kwargs):
    afields = a.fields
    bfields = b.fields

    assert len(afields) == len(bfields), \
        f'records have mismatched field counts: {len(afields)} != {len(bfields)}\n{a.names!r} != {b.names!r}\n{_fmt_path(path)}'  # noqa: E501

    if not check_record_order:
        afields = sorted(afields)
        bfields = sorted(bfields)

    if path is None:
        path = ()
    for n, ((aname, afield), (bname, bfield)) in enumerate(
            zip(afields, bfields)):

        assert aname == bname, \
            f'record field name at position {n} does not match: {aname!r} != {bname!r}\n{_fmt_path(path)}'  # noqa: E501

        assert_dshape_equal(
            afield,
            bfield,
            path=path + (f'[{repr(aname)}]',),
            check_record_order=check_record_order,
            **kwargs
        )


@dispatch(Tuple, Tuple)
def assert_dshape_equal(a, b, path=None, **kwargs):
    assert len(a.dshapes) == len(b.dshapes), \
        f'tuples have mismatched field counts: {len(a.dshapes)} != {len(b.dshapes)}\n{a!r} != {b!r}\n{_fmt_path(path)}'  # noqa: E501

    if path is None:
        path = ()
    path += '.dshapes',
    for n, (ashape, bshape) in enumerate(zip(a.dshapes, b.dshapes)):
        assert_dshape_equal(
            ashape,
            bshape,
            path=path + (f'[{n}]',),
            **kwargs
        )


@dispatch(Function, Function)
def assert_dshape_equal(a, b, path=None, **kwargs):
    assert len(a.argtypes) == len(b.argtypes),\
        f'functions have different arities: {len(a.argtypes)} != {len(b.argtypes)}\n{a!r} != {b!r}\n{_fmt_path(path)}'  # noqa: E501

    if path is None:
        path = ()
    for n, (aarg, barg) in enumerate(zip(a.argtypes, b.argtypes)):
        assert_dshape_equal(
            aarg,
            barg,
            path=path + (f'.argtypes[{n}]',), **kwargs
        )
    assert_dshape_equal(
        a.restype,
        b.restype,
        path=path + ('.restype',),
        **kwargs
    )
