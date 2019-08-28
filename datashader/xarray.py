from __future__ import absolute_import
from datashader.glyphs.quadmesh import _QuadMeshLike
from datashader.pandas import default
from .core import bypixel
import xarray as xr
from .utils import Dispatcher


glyph_dispatch = Dispatcher()


@bypixel.pipeline.register(xr.Dataset)
def xarray_pipeline(df, schema, canvas, glyph, summary):
    return glyph_dispatch(glyph, df, schema, canvas, summary)


# Default to default pandas implementation
glyph_dispatch.register(_QuadMeshLike)(default)
