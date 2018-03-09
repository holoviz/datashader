from __future__ import absolute_import

try:
    from param.version import Version
    __version__ = str(Version(fpath=__file__, archive_commit="$Format:%h$", reponame="datashader"))
    del Version
except:
    import os, json
    __version__ = json.load(open(os.path.join(os.path.split(__file__)[0],
                                              '.version'), 'r'))['version_string']
    del os
    del json

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

    import os
    from shutil import copytree, ignore_patterns

    candidates = [os.path.join(__path__[0], '../examples'),
                  os.path.join(__path__[0], '../../../../share/datashader-examples')]

    for source in candidates:
        if os.path.exists(source):
            copytree(source, path, ignore=ignore_patterns('data', '.ipynb_checkpoints', '*.pyc', '*~'))
            if verbose:
                print("%s copied to %s" % (source, path))
            break
