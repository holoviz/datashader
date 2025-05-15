from __future__ import annotations
from datashader.data_libraries.pandas import default
from datashader.core import bypixel
from .._dependencies import cudf


def cudf_pipeline(df, schema, canvas, glyph, summary, *, antialias=False):
    return default(glyph, df, schema, canvas, summary, antialias=antialias, cuda=True)


bypixel.pipeline.lazy_register(cudf, "DataFrame", cudf_pipeline)
