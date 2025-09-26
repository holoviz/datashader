from __future__ import annotations

from packaging.version import Version

from .__version import __version__  # noqa: F401

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
