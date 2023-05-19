from __future__ import annotations
from datashader.glyphs.quadmesh import _QuadMeshLike
from datashader.data_libraries.pandas import default
from datashader.core import bypixel
import xarray as xr
from datashader.utils import Dispatcher


try:
    import cupy
except Exception:
    cupy = None

glyph_dispatch = Dispatcher()


@bypixel.pipeline.register(xr.Dataset)
def xarray_pipeline(xr_ds, schema, canvas, glyph, summary, *, antialias=False):
    cuda = cupy and isinstance(xr_ds[glyph.name].data, cupy.ndarray)
    if not xr_ds.chunks:
        return glyph_dispatch(
            glyph, xr_ds, schema, canvas, summary, antialias=antialias, cuda=cuda)
    else:
        from datashader.data_libraries.dask_xarray import dask_xarray_pipeline
        return dask_xarray_pipeline(
            glyph, xr_ds, schema, canvas, summary, antialias=antialias, cuda=cuda)


# Default to default pandas implementation
glyph_dispatch.register(_QuadMeshLike)(default)
