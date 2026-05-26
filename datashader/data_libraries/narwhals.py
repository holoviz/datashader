from __future__ import annotations

import narwhals as nw

from datashader.core import bypixel
from datashader.data_libraries.pandas import default

__all__ = ()


@bypixel.pipeline.register(nw.DataFrame)
def narwhals_pipeline(df, schema, canvas, glyph, summary, *, antialias=False):
    df = df.to_pandas()
    return default(glyph, df, schema, canvas, summary, antialias=antialias)
