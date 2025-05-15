from __future__ import annotations
from datashader.data_libraries.dask import dask_pipeline
from datashader.core import bypixel
from .._dependencies import dask_cudf


def dask_cudf_pipeline(df, schema, canvas, glyph, summary, *, antialias=False):
    return dask_pipeline(df, schema, canvas, glyph, summary, antialias=antialias, cuda=True)


bypixel.pipeline.lazy_register(dask_cudf, "DataFrame", dask_cudf_pipeline)
