from __future__ import absolute_import

__version__ = '0.2.0'

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
