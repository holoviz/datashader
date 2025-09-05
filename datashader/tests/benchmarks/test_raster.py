import pytest
import numpy as np
import xarray as xr
import datashader as ds

from . import array_fixtures, cupy


@array_fixtures
def raster_data(request):
    size, array_module = request.param
    if array_module is cupy:
        pytest.skip("not currently supported")
    data = xr.DataArray(
        array_module.random.random((size, size)),
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
