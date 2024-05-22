from __future__ import annotations
from contextlib import suppress
from datashader.data_libraries.pandas import default
from datashader.core import bypixel


def cudf_pipeline(df, schema, canvas, glyph, summary, *, antialias=False):
    return default(glyph, df, schema, canvas, summary, antialias=antialias, cuda=True)


with suppress(ImportError):
    import cudf

    cudf_pipeline = bypixel.pipeline.register(cudf.DataFrame)(cudf_pipeline)
