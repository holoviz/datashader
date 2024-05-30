from datashader.compiler import compile_components
from datashader.utils import Dispatcher
from datashader.glyphs.line import LinesXarrayCommonX
from datashader.glyphs.quadmesh import (
    QuadMeshRaster, QuadMeshRectilinear, QuadMeshCurvilinear, build_scale_translate
)
from datashader.utils import apply
import dask
import numpy as np
import xarray as xr
from dask.base import tokenize, compute
from dask.array.overlap import overlap
dask_glyph_dispatch = Dispatcher()


def dask_xarray_pipeline(glyph, xr_ds, schema, canvas, summary, *, antialias=False, cuda=False):
    dsk, name = dask_glyph_dispatch(
        glyph, xr_ds, schema, canvas, summary, antialias=antialias, cuda=cuda)

    # Get user configured scheduler (if any), or fall back to default
    # scheduler for dask DataFrame
    scheduler = dask.base.get_scheduler() or xr_ds.__dask_scheduler__
    keys = xr_ds.__dask_keys__()
    optimize = xr_ds.__dask_optimize__
    graph = xr_ds.__dask_graph__()

    dsk.update(optimize(graph, keys))

    return scheduler(dsk, name)


def shape_bounds_st_and_axis(xr_ds, canvas, glyph):
    if not canvas.x_range or not canvas.y_range:
        x_extents, y_extents = glyph.compute_bounds_dask(xr_ds)
    else:
        x_extents, y_extents = None, None

    x_range = canvas.x_range or x_extents
    y_range = canvas.y_range or y_extents
    x_min, x_max, y_min, y_max = bounds = compute(*(x_range + y_range))
    x_range, y_range = (x_min, x_max), (y_min, y_max)

    width = canvas.plot_width
    height = canvas.plot_height

    x_st = canvas.x_axis.compute_scale_and_translate(x_range, width)
    y_st = canvas.y_axis.compute_scale_and_translate(y_range, height)
    st = x_st + y_st
    shape = (height, width)

    x_axis = canvas.x_axis.compute_index(x_st, width)
    y_axis = canvas.y_axis.compute_index(y_st, height)
    axis = dict([(glyph.x_label, x_axis), (glyph.y_label, y_axis)])

    return shape, bounds, st, axis


def dask_rectilinear(glyph, xr_ds, schema, canvas, summary, *, antialias=False, cuda=False):
    shape, bounds, st, axis = shape_bounds_st_and_axis(xr_ds, canvas, glyph)

    # Compile functions
    create, info, append, combine, finalize, antialias_stage_2, antialias_stage_2_funcs, _ = \
        compile_components(summary, schema, glyph, antialias=antialias, cuda=cuda, partitioned=True)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(
        x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs)
    x_range = bounds[:2]
    y_range = bounds[2:]

    # Build chunk indices for coordinates
    chunk_inds = {}
    for k, chunks in xr_ds.chunks.items():
        chunk_inds[k] = [0] + list(np.cumsum(chunks))

    x_name = glyph.x
    y_name = glyph.y
    coords = xr_ds[glyph.name].coords
    coord_dims = list(coords.dims)
    xdim_ind = coord_dims.index(x_name)
    ydim_ind = coord_dims.index(y_name)

    var_name = list(xr_ds.data_vars.keys())[0]

    # Compute interval breaks
    xs = xr_ds[x_name].values
    ys = xr_ds[y_name].values
    x_breaks = glyph.infer_interval_breaks(xs)
    y_breaks = glyph.infer_interval_breaks(ys)

    def chunk(np_arr, *inds):
        # Reconstruct dataset for chunk from numpy array and chunk indices
        chunk_coords_list = []
        # for i, (coord_name, coord_vals) in enumerate(coords.items()):
        for i, coord_name in enumerate(coords.dims):
            chunk_number = inds[i]
            coord_slice = slice(
                chunk_inds[coord_name][chunk_number],
                chunk_inds[coord_name][chunk_number + 1]
            )
            chunk_coords_list.append([coord_name, coords[coord_name][coord_slice]])

        chunk_coords = dict(chunk_coords_list)
        chunk_ds = xr.DataArray(
            np_arr, coords=chunk_coords, dims=coord_dims, name=var_name
        ).to_dataset()

        # Compute chunk x/y breaks
        x_chunk_number = inds[xdim_ind]
        x_breaks_slice = slice(
            chunk_inds[x_name][x_chunk_number],
            chunk_inds[x_name][x_chunk_number + 1] + 1
        )
        x_breaks_chunk = x_breaks[x_breaks_slice]

        y_chunk_number = inds[ydim_ind]
        y_breaks_slice = slice(
            chunk_inds[y_name][y_chunk_number],
            chunk_inds[y_name][y_chunk_number + 1] + 1
        )
        y_breaks_chunk = y_breaks[y_breaks_slice]

        # Initialize aggregation buffers
        aggs = create(shape)

        # Perform aggregation
        extend(aggs, chunk_ds, st, bounds,
               x_breaks=x_breaks_chunk, y_breaks=y_breaks_chunk)
        return aggs

    name = tokenize(xr_ds.__dask_tokenize__(), canvas, glyph, summary)
    keys = [k for row in xr_ds.__dask_keys__()[0] for k in row]
    keys2 = [(name, i) for i in range(len(keys))]
    dsk = dict((k2, (chunk, k, k[1], k[2])) for (k2, k) in zip(keys2, keys))
    dsk[name] = (apply, finalize, [(combine, keys2)],
                 dict(cuda=cuda, coords=axis, dims=[glyph.y_label, glyph.x_label],
                      attrs=dict(x_range=x_range, y_range=y_range)))
    return dsk, name


def dask_raster(glyph, xr_ds, schema, canvas, summary, *, antialias=False, cuda=False):
    shape, bounds, st, axis = shape_bounds_st_and_axis(xr_ds, canvas, glyph)

    # Compile functions
    create, info, append, combine, finalize, antialias_stage_2, antialias_stage_2_funcs, _ = \
        compile_components(summary, schema, glyph, antialias=antialias, cuda=cuda, partitioned=True)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(
        x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs)
    x_range = bounds[:2]
    y_range = bounds[2:]

    # Build chunk indices for coordinates
    chunk_inds = {}
    for k, chunks in xr_ds.chunks.items():
        chunk_inds[k] = [0] + list(np.cumsum(chunks))

    x_name = glyph.x
    y_name = glyph.y

    coords = xr_ds[glyph.name].coords

    coord_dims = list(coords.dims)
    xdim_ind = coord_dims.index(x_name)
    ydim_ind = coord_dims.index(y_name)
    var_name = list(xr_ds.data_vars.keys())[0]

    # Pre-compute bin sizes. We do this here to handle length-1 chunks
    src_x0, src_x1 = glyph. _compute_bounds_from_1d_centers(
        xr_ds, x_name, maybe_expand=False, orient=False
    )
    src_y0, src_y1 = glyph._compute_bounds_from_1d_centers(
        xr_ds, y_name, maybe_expand=False, orient=False
    )
    xbinsize = abs(float(xr_ds[x_name][1] - xr_ds[x_name][0]))
    ybinsize = abs(float(xr_ds[y_name][1] - xr_ds[y_name][0]))

    # Compute scale/translate
    out_h, out_w = shape
    src_h, src_w = [xr_ds[glyph.name].shape[i] for i in [ydim_ind, xdim_ind]]
    out_x0, out_x1, out_y0, out_y1 = bounds
    scale_y, translate_y = build_scale_translate(
        out_h, out_y0, out_y1, src_h, src_y0, src_y1
    )

    scale_x, translate_x = build_scale_translate(
        out_w, out_x0, out_x1, src_w, src_x0, src_x1
    )

    def chunk(np_arr, *inds):
        # Reconstruct dataset for chunk from numpy array and chunk indices
        chunk_coords_list = []
        for i, coord_name in enumerate(coords.dims):
            chunk_number = inds[i]
            coord_slice = slice(
                chunk_inds[coord_name][chunk_number],
                chunk_inds[coord_name][chunk_number + 1]
            )
            chunk_coords_list.append([coord_name, coords[coord_name][coord_slice]])

        chunk_coords = dict(chunk_coords_list)
        chunk_ds = xr.DataArray(
            np_arr, coords=chunk_coords, dims=coord_dims, name=var_name
        ).to_dataset()

        # Compute offsets
        x_chunk_number = inds[xdim_ind]
        offset_x = chunk_inds[x_name][x_chunk_number]

        y_chunk_number = inds[ydim_ind]
        offset_y = chunk_inds[y_name][y_chunk_number]

        # Initialize aggregation buffers
        aggs = create(shape)

        # Perform aggregation
        extend(aggs, chunk_ds, st, bounds,
               scale_x=scale_x, scale_y=scale_y,
               translate_x=translate_x, translate_y=translate_y,
               offset_x=offset_x, offset_y=offset_y,
               src_xbinsize=xbinsize, src_ybinsize=ybinsize)

        return aggs

    name = tokenize(xr_ds.__dask_tokenize__(), canvas, glyph, summary)
    keys = [k for row in xr_ds.__dask_keys__()[0] for k in row]
    keys2 = [(name, i) for i in range(len(keys))]
    dsk = dict((k2, (chunk, k, k[1], k[2])) for (k2, k) in zip(keys2, keys))
    dsk[name] = (apply, finalize, [(combine, keys2)],
                 dict(cuda=cuda, coords=axis, dims=[glyph.y_label, glyph.x_label],
                      attrs=dict(x_range=x_range, y_range=y_range)))
    return dsk, name


def dask_curvilinear(glyph, xr_ds, schema, canvas, summary, *, antialias=False, cuda=False):
    shape, bounds, st, axis = shape_bounds_st_and_axis(xr_ds, canvas, glyph)

    # Compile functions
    create, info, append, combine, finalize, antialias_stage_2, antialias_stage_2_funcs, _ = \
        compile_components(summary, schema, glyph, antialias=antialias, cuda=cuda, partitioned=True)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(
        x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs)
    x_range = bounds[:2]
    y_range = bounds[2:]

    x_coord_name = glyph.x
    y_coord_name = glyph.y
    z_name = glyph.name

    data_dim_names = list(xr_ds[z_name].dims)
    x_coord_dim_names = list(xr_ds[x_coord_name].dims)
    y_coord_dim_names = list(xr_ds[y_coord_name].dims)
    zs = xr_ds[z_name].data
    x_centers = xr_ds[glyph.x].data
    y_centers = xr_ds[glyph.y].data

    var_name = list(xr_ds.data_vars.keys())[0]

    # Validate coordinates
    err_msg = (
        "DataArray {name} is backed by a Dask array, \n"
        "but coordinate {coord} is not backed by a Dask array with identical \n"
        "dimension order and chunks"
    )
    if (not isinstance(x_centers, dask.array.Array) or
            xr_ds[glyph.name].dims != xr_ds[glyph.x].dims or
            xr_ds[glyph.name].chunks != xr_ds[glyph.x].chunks):
        raise ValueError(err_msg.format(name=glyph.name, coord=glyph.x))

    if (not isinstance(y_centers, dask.array.Array) or
            xr_ds[glyph.name].dims != xr_ds[glyph.y].dims or
            xr_ds[glyph.name].chunks != xr_ds[glyph.y].chunks):
        raise ValueError(err_msg.format(name=glyph.name, coord=glyph.y))

    # Make sure coordinates are floats so that overlap with nan will behave properly
    if x_centers.dtype.kind != 'f':
        x_centers = x_centers.astype(np.float64)
    if y_centers.dtype.kind != 'f':
        y_centers = y_centers.astype(np.float64)

    x_overlapped_centers = overlap(x_centers, depth=1, boundary=np.nan)
    y_overlapped_centers = overlap(y_centers, depth=1, boundary=np.nan)

    def chunk(np_zs, np_x_centers, np_y_centers):

        # Handle boundaries that have nothing to overlap with
        for centers in [np_x_centers, np_y_centers]:
            if np.isnan(centers[0, :]).all():
                centers[0, :] = centers[1, :] - (centers[2, :] - centers[1, :])
            if np.isnan(centers[-1, :]).all():
                centers[-1, :] = centers[-2, :] + (centers[-2, :] - centers[-3, :])
            if np.isnan(centers[:, 0]).all():
                centers[:, 0] = centers[:, 1] - (centers[:, 2] - centers[:, 1])
            if np.isnan(centers[:, -1]).all():
                centers[:, -1] = centers[:, -2] + (centers[:, -2] - centers[:, -3])

        # compute interval breaks
        x_breaks_chunk = glyph.infer_interval_breaks(np_x_centers)
        y_breaks_chunk = glyph.infer_interval_breaks(np_y_centers)

        # trim breaks
        x_breaks_chunk = x_breaks_chunk[1:-1, 1:-1]
        y_breaks_chunk = y_breaks_chunk[1:-1, 1:-1]

        # Reconstruct dataset for chunk from numpy array and chunk indices
        chunk_coords = {
            x_coord_name: (x_coord_dim_names, np_x_centers[1:-1, 1:-1]),
            y_coord_name: (y_coord_dim_names, np_y_centers[1:-1, 1:-1]),
        }
        chunk_ds = xr.DataArray(
            np_zs, coords=chunk_coords, dims=data_dim_names, name=var_name
        ).to_dataset()

        # Initialize aggregation buffers
        aggs = create(shape)

        # Perform aggregation
        extend(aggs, chunk_ds, st, bounds,
               x_breaks=x_breaks_chunk, y_breaks=y_breaks_chunk)
        return aggs

    result_name = tokenize(xr_ds.__dask_tokenize__(), canvas, glyph, summary)

    z_keys = [k for row in zs.__dask_keys__() for k in row]
    x_overlap_keys = [k for row in x_overlapped_centers.__dask_keys__() for k in row]
    y_overlap_keys = [k for row in y_overlapped_centers.__dask_keys__() for k in row]

    result_keys = [(result_name, i) for i in range(len(z_keys))]

    dsk = dict(
        (res_k, (chunk, z_k, x_k, y_k))
        for (res_k, z_k, x_k, y_k) in zip(
            result_keys, z_keys, x_overlap_keys, y_overlap_keys
        )
    )

    dsk[result_name] = (
        apply, finalize, [(combine, result_keys)],
        dict(cuda=cuda, coords=axis, dims=[glyph.y_label, glyph.x_label],
             attrs=dict(x_range=x_range, y_range=y_range))
    )

    # Add x/y coord tasks to task graph
    dsk.update(x_overlapped_centers.dask)
    dsk.update(y_overlapped_centers.dask)

    return dsk, result_name


def dask_xarray_lines(
    glyph: LinesXarrayCommonX, xr_ds: xr.Dataset, schema, canvas, summary,
    *, antialias=False, cuda=False,
):
    shape, bounds, st, axis = shape_bounds_st_and_axis(xr_ds, canvas, glyph)

    # Compile functions
    create, info, append, combine, finalize, antialias_stage_2, antialias_stage_2_funcs, \
        column_names = compile_components(summary, schema, glyph, antialias=antialias, cuda=cuda,
                                          partitioned=True)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(
        x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs)
    x_range = bounds[:2]
    y_range = bounds[2:]

    x_name = glyph.x
    x_dim_index = glyph.x_dim_index
    other_dim_index = 1 - x_dim_index
    other_dim_name = xr_ds[glyph.y].coords.dims[other_dim_index]
    xs = xr_ds[x_name]

    # Build chunk offsets for coordinates
    chunk_offsets = {}
    for k, chunks in xr_ds.chunks.items():
        chunk_offsets[k] = [0] + list(np.cumsum(chunks))

    partitioned = True
    uses_row_index = summary.uses_row_index(cuda, partitioned)

    def chunk(np_array, *chunk_indices):
        aggs = create(shape)

        start_x_index = chunk_offsets[x_name][chunk_indices[x_dim_index]]
        end_x_index = start_x_index + np_array.shape[x_dim_index]
        x = xs[start_x_index:end_x_index].values

        start_other_index = chunk_offsets[other_dim_name][chunk_indices[other_dim_index]]
        end_other_index = start_other_index + np_array.shape[other_dim_index]

        data_vars = dict(
            name=(("x", other_dim_name) if x_dim_index == 0 else (other_dim_name, "x"), np_array),
        )
        # Other required columns are chunked in the other_dim
        for column_name in column_names:
            values = xr_ds[column_name][start_other_index:end_other_index].values
            data_vars[column_name] = (other_dim_name, values)

        chunk_ds = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                x=("x", x),
                other_dim_name=(other_dim_name, np.arange(start_other_index, end_other_index)),
            ),
        )

        if uses_row_index:
            row_offset = start_other_index
            chunk_ds.attrs["_datashader_row_offset"] = row_offset
            chunk_ds.attrs["_datashader_row_length"] = end_other_index - start_other_index

        extend(aggs, chunk_ds, st, bounds)
        return aggs

    name = tokenize(xr_ds.__dask_tokenize__(), canvas, glyph, summary)
    keys = [k for row in xr_ds.__dask_keys__()[0] for k in row]
    keys2 = [(name, i) for i in range(len(keys))]
    dsk = dict((k2, (chunk, k, k[1], k[2])) for (k2, k) in zip(keys2, keys))
    dsk[name] = (apply, finalize, [(combine, keys2)],
                 dict(cuda=cuda, coords=axis, dims=[glyph.y_label, glyph.x_label],
                      attrs=dict(x_range=x_range, y_range=y_range)))
    return dsk, name


dask_glyph_dispatch.register(QuadMeshRectilinear)(dask_rectilinear)
dask_glyph_dispatch.register(QuadMeshRaster)(dask_raster)
dask_glyph_dispatch.register(QuadMeshCurvilinear)(dask_curvilinear)
dask_glyph_dispatch.register(LinesXarrayCommonX)(dask_xarray_lines)
