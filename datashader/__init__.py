from __future__ import absolute_import

__version__ = '0.0.1'

from .core import Canvas
from .reductions import (count, sum, min, max, mean, std, var, count_cat,
                         summary)
from .glyphs import Point
from .pipeline import Pipeline

# Needed to build the backend dispatch
from .pandas import *
try:
    from .dask import *
except ImportError:
    pass
