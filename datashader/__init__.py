from __future__ import absolute_import

import param
__version__ = str(param.version.Version(fpath=__file__, archive_commit="$Format:%h$",reponame="datashader"))

from .core import Canvas                                 # noqa (API import)
from .reductions import *                                # noqa (API import)
from .glyphs import Point                                # noqa (API import)
from .pipeline import Pipeline                           # noqa (API import)
from . import transfer_functions as tf                   # noqa (API import)

from . import pandas                         # noqa (build backend dispatch)
try:
    from . import dask                       # noqa (build backend dispatch)
except ImportError:
    pass

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
