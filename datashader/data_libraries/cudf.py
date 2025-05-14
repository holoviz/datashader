from __future__ import annotations
from datashader.data_libraries.pandas import default
from datashader.core import bypixel
from .._dependencies import cudf, register_import_hook


def cudf_pipeline(df, schema, canvas, glyph, summary, *, antialias=False):
    return default(glyph, df, schema, canvas, summary, antialias=antialias, cuda=True)


register_import_hook(cudf, lambda: bypixel.pipeline.register(cudf.DataFrame)(cudf_pipeline))
