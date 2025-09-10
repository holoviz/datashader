import pytest
import numpy as np
import xarray as xr
import datashader as ds


DATA_SIZES = (256, 512, 1024, 2048, 4096, 8192)
CANVAS_SIZE = (1024, 1024)


@pytest.fixture(params=DATA_SIZES)
def raster_dask_data(request, rng):
    da = pytest.importorskip("dask.array")
    size = request.param
    data = xr.DataArray(
        da.random.random((size, size)),
        dims=["x", "y"],
        coords={"x": np.arange(size), "y": np.arange(size)},
        name="raster_data",
    )

    return data


@pytest.mark.benchmark(group="raster")
def test_dask_raster(benchmark, raster_dask_data):
    # Benchmark for https://github.com/holoviz/datashader/pull/1448
    def func():
        cvs = ds.Canvas(*CANVAS_SIZE)
        arr = cvs.raster(raster_dask_data)
        return arr.compute()

    benchmark(func)
