from __future__ import annotations

from packaging.version import Version

from .__version import __version__  # noqa: F401

from .core import Canvas                                 # noqa (API import)
from .reductions import *                                # noqa (API import)
from .glyphs import Point                                # noqa (API import)
from .pipeline import Pipeline                           # noqa (API import)
from . import transfer_functions as tf                   # noqa (API import)
from . import data_libraries                             # noqa (API import)

# Make RaggedArray pandas extension array available for
# pandas >= 0.24.0 is installed
from pandas import __version__ as pandas_version
if Version(pandas_version) >= Version('0.24.0'):
    from . import datatypes  # noqa (API import)

# make pyct's example/data commands available if possible
from functools import partial, wraps


def _warn_pyct_deprecated(stacklevel=2):
    import warnings

    warnings.warn(
        "The 'fetch_data()', 'copy_examples()', and 'examples()' functions are "
        "deprecated since version 0.19 and will be removed in version 0.20. "
        "For downloading sample datasets, use 'hvsampledata' instead. "
        "For example: `hvsampledata.nyc_taxi_remote('pandas')`.",
        category=FutureWarning,
        stacklevel=stacklevel,
    )


def _deprecated_pyct_wrapper(func):
    """Wrapper to add deprecation warning to pyct functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        _warn_pyct_deprecated(stacklevel=3)
        return func(*args, **kwargs)
    return wrapper


try:
    from pyct.cmd import copy_examples as _copy, fetch_data as _fetch, examples as _examples
    copy_examples = _deprecated_pyct_wrapper(partial(_copy, 'datashader'))
    fetch_data = _deprecated_pyct_wrapper(partial(_fetch, 'datashader'))
    examples = _deprecated_pyct_wrapper(partial(_examples, 'datashader'))
except ImportError:
    def _missing_cmd(*args,**kw):
        return("install pyct to enable this command (e.g. `conda install pyct or "
               "`pip install pyct[cmd]`)")
    _copy = _fetch = _examples = _missing_cmd
    def err():
        raise ValueError(_missing_cmd())
    fetch_data = copy_examples = examples = err
del partial, _examples, _copy, _fetch
