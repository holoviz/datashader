import pytest
import numpy as np
import xarray as xr
import datashader as ds

sizes = [256, 512, 1024, 2048, 4096, 8192]
CANVAS_SIZE = (1024, 1024)
rng = np.random.default_rng()

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

    return data, (west, east), (south, north)


@pytest.mark.benchmark(group="quadmesh")
def test_quadmesh_curvilinear(benchmark, quadmesh_data):
    def func():
        data, x_range, y_range = quadmesh_data
        lon_coord, lat_coord = xr.broadcast(data.x, data.y)
        data = data.assign_coords({"lon": lon_coord, "lat": lat_coord})
        cvs = ds.Canvas(*CANVAS_SIZE, x_range=x_range, y_range=y_range)
        quadmesh = cvs.quadmesh(data.transpose("y", "x"), x="lon", y="lat")
        return quadmesh.compute()

    benchmark(func)

@pytest.mark.benchmark(group="quadmesh")
def test_quadmesh_raster(benchmark, quadmesh_data):
    data, x_range, y_range = quadmesh_data
    data = data.swap_dims({"y": "lat", "x": "lon"})

    def func():
        cvs = ds.Canvas(*CANVAS_SIZE, x_range=x_range, y_range=y_range)
        quadmesh = cvs.quadmesh(data.transpose("lat", "lon"), x="lon", y="lat")
        return quadmesh.compute()

    benchmark(func)


@pytest.mark.benchmark(group="quadmesh")
def test_quadmesh_rectilinear(benchmark, quadmesh_data):
    data, x_range, y_range = quadmesh_data
    dy = (y_range[1] - y_range[0]) / data.sizes["y"]
    deltas = rng.uniform(-dy/4, dy/4, data.sizes["y"])
    data["y"] = ("y", data.y.data + deltas)
    data = data.swap_dims({"y": "lat", "x": "lon"})

    def func():
        cvs = ds.Canvas(*CANVAS_SIZE, x_range=x_range, y_range=y_range)
        quadmesh = cvs.quadmesh(data.transpose("lat", "lon"), x="lon", y="lat")
        return quadmesh.compute()

    benchmark(func)
