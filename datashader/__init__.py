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


def test():
    """Run the datashader test suite."""
    import os
    try:
        import pytest
    except ImportError:
        import sys
        sys.stderr.write("You need to install py.test to run tests.\n\n")
        raise
    pytest.main([os.path.dirname(__file__)])


def examples(path='datashader-examples', verbose=False):
    """
    Copies the examples to the supplied path.
    """
    import os, warnings, distutils.dir_util
    source = os.path.join(os.path.dirname(__file__),"examples")
    path = os.path.abspath(path)
    if os.path.exists(path):
        warnings.warn("Path %s already exists; will not overwrite newer files."%path)
    distutils.dir_util.copy_tree(source, path, verbose=verbose)
    print("Installed examples at %s"%path)
