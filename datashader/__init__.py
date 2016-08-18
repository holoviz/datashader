from __future__ import absolute_import

__version__ = '0.3.2'

from .core import Canvas
from .reductions import (count, any, sum, min, max, mean, std, var, count_cat,
                         summary)
from .glyphs import Point
from .pipeline import Pipeline

# Needed to build the backend dispatch
from .pandas import *
try:
    from .dask import *
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
    pytest.main(os.path.dirname(__file__))


def examples(path='datashader-examples', verbose=False):
    """
    Copies the examples to the supplied path.
    """

    import os, glob
    from shutil import copytree, ignore_patterns

    candidates = [os.path.join(__path__[0], '../examples'),
                  os.path.join(__path__[0], '../../../../share/datashader-examples')]

    for source in candidates:
        if os.path.exists(source):
            copytree(source, path, ignore=ignore_patterns('data','.ipynb_checkpoints','*.pyc','*~'))
            if verbose:
                print("%s copied to %s" % (source, path))
            break
