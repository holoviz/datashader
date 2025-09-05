import pytest
import numpy as np
import xarray as xr
import datashader as ds


sizes = [256, 512, 1024, 2048, 4096, 8192]


@pytest.fixture(params=sizes)
def raster_data(request, rng):
    size = request.param
    data = xr.DataArray(
        rng.random((size, size)),
        dims=["x", "y"],
        coords={"x": np.arange(size), "y": np.arange(size)},
        name="raster_data",
    )

    return data


@pytest.mark.benchmark(group="raster")
def test_raster(benchmark, raster_data):
    def func():
        cvs = ds.Canvas(plot_height=300, plot_width=300)
        arr = cvs.raster(raster_data)
        return arr.compute()

    benchmark(func)
