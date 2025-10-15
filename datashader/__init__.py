from __future__ import annotations

import warnings
from contextlib import suppress
from packaging.version import Version

from .__version import __version__  # noqa: F401

from .core import Canvas                                 # noqa (API import)
from .reductions import *                                # noqa (API import)
from .glyphs import Point                                # noqa (API import)
from .pipeline import Pipeline                           # noqa (API import)
from . import transfer_functions as tf                   # noqa (API import)
from . import data_libraries                             # noqa (API import)

with suppress(ImportError):
    import pyct  # noqa: F401

    warnings.warn(
        "The 'pyct' package bundled as a datashader dependency is deprecated since version 0.19 "
        "and will be removed in version 0.20. For downloading sample datasets, "
        "prefer using 'hvsampledata' (for example: "
        "`hvsampledata.nyc_taxi_remote('pandas')`).",
        category=FutureWarning,
        stacklevel=2,
    )

# Make RaggedArray pandas extension array available for
# pandas >= 0.24.0 is installed
from pandas import __version__ as pandas_version
if Version(pandas_version) >= Version('0.24.0'):
    from . import datatypes  # noqa (API import)

# make pyct's example commands available if possible
from functools import partial
try:
    from pyct.cmd import copy_examples as _copy, examples as _examples
    copy_examples = partial(_copy,'datashader')
    examples = partial(_examples,'datashader')
except ImportError:
    def _missing_cmd(*args,**kw):
        return("install pyct to enable this command (e.g. `conda install pyct or "
               "`pip install pyct[cmd]`)")
    _copy = _examples = _missing_cmd
    def err():
        raise ValueError(_missing_cmd())
    copy_examples = examples = err
del partial, _examples, _copy
