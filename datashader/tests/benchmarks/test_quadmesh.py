import pytest
import numpy as np
import xarray as xr
import datashader as ds

from . import array_params, da


@pytest.fixture(params=array_params)
def quadmesh_data(request):
    """Create test data array for quadmesh benchmarking"""
    size, array_module = request.param
    west = 3125000.0
    south = 3250000.0
    east = 4250000.0
    north = 4375000.0

    data = xr.DataArray(
        array_module.random.random((size, size)),
        dims=("x", "y"),
        coords={
            "lon": ("x", np.linspace(3123580.0, 4250380.0, size)),
            "lat": ("y", np.linspace(4376200.0, 3249400.0, size)),
        },
        name="benchmark_data",
    )

    data = data.isel(y=slice(None, None, -1))

    lon_coord, lat_coord = xr.broadcast(data.x, data.y)
    if array_module is da:
        chunks_dict = dict(zip(data.dims, data.chunks))
        lon_coord = lon_coord.chunk(chunks_dict)
        lat_coord = lat_coord.chunk(chunks_dict)

    data = data.assign_coords({"lon": lon_coord, "lat": lat_coord})

    return data, (west, east), (south, north)


@pytest.mark.benchmark(group="quadmesh")
def test_quadmesh(benchmark, quadmesh_data):
    """Benchmark quadmesh operation"""
    data, x_range, y_range = quadmesh_data
    cvs = ds.Canvas(256, 256, x_range=x_range, y_range=y_range)

    benchmark(cvs.quadmesh, data.transpose("y", "x"), x="lon", y="lat")
