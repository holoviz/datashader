from __future__ import absolute_import

__version__ = '0.0.1'

from .expr import Canvas

# Needed to build the backend dispatch
from . import backends
