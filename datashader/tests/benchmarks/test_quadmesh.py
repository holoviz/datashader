import pytest
import numpy as np
import xarray as xr
import datashader as ds

sizes = [256, 512, 1024, 2048, 4096, 8192]


@pytest.fixture(params=sizes)
def quadmesh_data(request, rng):
    size = request.param
    west = 3125000.0
    south = 3250000.0
    east = 4250000.0
    north = 4375000.0

    data = xr.DataArray(
        rng.random((size, size)),
        dims=("x", "y"),
        coords={
            "lon": ("x", np.linspace(3123580.0, 4250380.0, size)),
            "lat": ("y", np.linspace(4376200.0, 3249400.0, size)),
        },
        name="benchmark_data",
    )
    data = data.isel(y=slice(None, None, -1))
    lon_coord, lat_coord = xr.broadcast(data.x, data.y)
    data = data.assign_coords({"lon": lon_coord, "lat": lat_coord})

    return data, (west, east), (south, north)


@pytest.mark.benchmark(group="quadmesh")
def test_quadmesh_curvilinear(benchmark, quadmesh_data):
    def func():
        data, x_range, y_range = quadmesh_data
        cvs = ds.Canvas(1024, 1024, x_range=x_range, y_range=y_range)
        quadmesh = cvs.quadmesh(data.transpose("y", "x"), x="lon", y="lat")
        return quadmesh.compute()

    benchmark(func)
