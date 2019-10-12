from __future__ import absolute_import
from datashader.data_libraries.dask import dask_pipeline
from datashader.core import bypixel
import dask_cudf


@bypixel.pipeline.register(dask_cudf.DataFrame)
def dask_cudf_pipeline(df, schema, canvas, glyph, summary):
    return dask_pipeline(df, schema, canvas, glyph, summary, cuda=True)
