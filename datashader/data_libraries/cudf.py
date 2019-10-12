from __future__ import absolute_import
from datashader.data_libraries.pandas import default
from datashader.core import bypixel
import cudf


@bypixel.pipeline.register(cudf.DataFrame)
def cudf_pipeline(df, schema, canvas, glyph, summary):
    return default(glyph, df, schema, canvas, summary, cuda=True)
