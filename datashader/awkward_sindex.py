from __future__ import annotations

import awkward as ak
import awkward_pandas as akpd
import numpy as np
import pandas as pd
from spatialpandas.spatialindex import HilbertRtree

from .utils import ngjit


ak.numba.register_and_check()  # Don't know if this is needed here


def awkward_sindex(df: pd.DataFrame, geometry: str):
    extension_array = df[geometry].array
    aa = extension_array._data

    n = len(aa)  # 1157859   # number of polygons
    bounds = np.empty((n, 4))

    x = aa[..., ::2]
    y = aa[..., 1::2]

    # Need to calc min/max across last dim 3 times to remove nested polygon possibilities.
    xmin = xmax = x
    ymin = ymax = y
    for _ in range(3):
        xmin = ak.min(xmin, axis=-1)
        xmax = ak.min(xmax, axis=-1)
        ymin = ak.min(ymin, axis=-1)
        ymax = ak.min(ymax, axis=-1)

    bounds[:, 0] = xmin
    bounds[:, 1] = ymin
    bounds[:, 2] = xmax
    bounds[:, 3] = ymax

    extension_array._sindex = HilbertRtree(bounds)  #, **kwargs)
