from __future__ import absolute_import

import param
__version__ = str(param.version.Version(fpath=__file__, archive_commit="$Format:%h$",reponame="datashader"))

from .core import Canvas                                 # noqa (API import)
from .reductions import (count, any, sum, min, max,      # noqa (API import)
                         mean, std, var, count_cat, summary)
from .glyphs import Point                                # noqa (API import)
from .pipeline import Pipeline                           # noqa (API import)
from . import transfer_functions as tf                   # noqa (API import)

from . import pandas                         # noqa (build backend dispatch)
try:
    from . import dask                       # noqa (build backend dispatch)
except ImportError:
    pass

# make pvutil's install_examples() and download_data() available if possible
from functools import partial
try:
    from pvutil.cmd import install_examples as _examples, download_data as _data
    install_examples = partial(_examples,'datashader')
    download_data = partial(_data,'datashader')
except ImportError:
    def _examples(*args,**kw): print("install examples package to enable this command (`conda install datashader-examples`)")
    _data = _examples
    def err(): raise ValueError(_data())   # noqa: _data is defined
    download_data = install_examples = err
del partial, _examples, _data
