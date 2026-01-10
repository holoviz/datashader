from __future__ import annotations
from datashader.glyphs.line import LinesXarrayCommonX
from datashader.glyphs.quadmesh import _QuadMeshLike
from datashader.data_libraries.pandas import default
from datashader.core import bypixel
import xarray as xr
from datashader.utils import Dispatcher
from datashader.compiler import compile_components


try:
    import cupy
except Exception:
    cupy = None

glyph_dispatch = Dispatcher()


@bypixel.pipeline.register(xr.Dataset)
def xarray_pipeline(xr_ds, schema, canvas, glyph, summary, *, antialias=False):
    cuda = False
    if cupy:
        if isinstance(glyph, LinesXarrayCommonX):
            cuda = isinstance(xr_ds[glyph.y].data, cupy.ndarray)
        else:
            cuda = isinstance(xr_ds[glyph.name].data, cupy.ndarray)

    if not xr_ds.chunks:
        return glyph_dispatch(
            glyph, xr_ds, schema, canvas, summary, antialias=antialias, cuda=cuda)
    else:
        from datashader.data_libraries.dask_xarray import dask_xarray_pipeline
        return dask_xarray_pipeline(
            glyph, xr_ds, schema, canvas, summary, antialias=antialias, cuda=cuda)


def _extract_third_dim(glyph, source):
    # Get the dimensions used by the x and y coordinates
    # For rectilinear, glyph.x/y are 1D dimension names (e.g., 'x', 'y')
    # For curvilinear, glyph.x/y are 2D coordinate names (e.g., 'lon', 'lat')
    x_dims = set(source.coords[glyph.x].dims) if glyph.x in source.coords else {glyph.x}
    y_dims = set(source.coords[glyph.y].dims) if glyph.y in source.coords else {glyph.y}
    coord_dims = x_dims | y_dims
    return next(iter(set(source.dims) - coord_dims), None)


@glyph_dispatch.register(_QuadMeshLike)
def quadmesh_default(glyph, source, schema, canvas, summary, *, antialias=False, cuda=False):
    third_dim = _extract_third_dim(glyph, source)
    if not third_dim:
        return default(glyph, source, schema, canvas, summary, antialias=antialias, cuda=cuda)

    create, info, append, _, finalize, antialias_stage_2, antialias_stage_2_funcs, _ = \
        compile_components(summary, schema, glyph, antialias=antialias, cuda=cuda,
                           partitioned=False)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(
        x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs)

    x_range = canvas.x_range or glyph.compute_x_bounds(source)
    y_range = canvas.y_range or glyph.compute_y_bounds(source)
    canvas.validate_ranges(x_range, y_range)

    width = canvas.plot_width
    height = canvas.plot_height

    x_st = canvas.x_axis.compute_scale_and_translate(x_range, width)
    y_st = canvas.y_axis.compute_scale_and_translate(y_range, height)

    x_axis = canvas.x_axis.compute_index(x_st, width)
    y_axis = canvas.y_axis.compute_index(y_st, height)

    bases = create((len(source.coords[third_dim]), height, width))

    extend(bases, source, x_st + y_st, x_range + y_range)

    return finalize(
        bases,
        cuda=cuda,
        coords=dict([
            (third_dim, source.coords[third_dim]),
            (glyph.x_label, x_axis),
            (glyph.y_label, y_axis),
        ]),
        dims=[third_dim, glyph.y_label, glyph.x_label],
        attrs=dict(x_range=x_range, y_range=y_range)
    )

# Default to default pandas implementation
glyph_dispatch.register(LinesXarrayCommonX)(default)
