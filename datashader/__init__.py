from __future__ import annotations

from packaging.version import Version

import param
__version__ = str(param.version.Version(fpath=__file__, archive_commit="$Format:%h$",reponame="datashader"))

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
from functools import partial
try:
    from pyct.cmd import copy_examples as _copy, fetch_data as _fetch, examples as _examples
    copy_examples = partial(_copy,'datashader')
    fetch_data = partial(_fetch,'datashader')
    examples = partial(_examples,'datashader')
except ImportError:
    def _missing_cmd(*args,**kw): return("install pyct to enable this command (e.g. `conda install pyct or `pip install pyct[cmd]`)")
    _copy = _fetch = _examples = _missing_cmd
    def err(): raise ValueError(_missing_cmd())
    fetch_data = copy_examples = examples = err
del partial, _examples, _copy, _fetch
