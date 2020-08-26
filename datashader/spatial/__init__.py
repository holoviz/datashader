from __future__ import absolute_import

import warnings

from datashader.spatial.points import read_parquet  # noqa (API import)

from ..utils import VisibleDeprecationWarning

warnings.warn(
    "The datashader.spatial module is deprecated as of version 0.11.0. "
    "The functionality it provided has migrated to the spatialpandas "
    "(github.com/holoviz/spatialpandas) and xarray-spatial "
    "(github.com/makepath/xarray-spatial) libraries.",
    VisibleDeprecationWarning
)
