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


def examples(path='.', verbose=False):
    """
    Copies the examples to the supplied path.
    """

    import os, glob
    from shutil import copyfile

    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(path)
        if verbose: print('Created directory %s' % path)
        
    notebook_glob = os.path.join(__path__[0], '..', 'examples', '*')
    notebooks = glob.glob(notebook_glob)
    
    for notebook in notebooks:
        nb_path = os.path.join(path, os.path.basename(notebook))
        copyfile(notebook, nb_path)
        if verbose: print("%s copied to %s" % (os.path.basename(notebook), path))
