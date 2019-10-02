from . import pandas, xarray  # noqa (API import)

try:
    import dask as _dask  # noqa (Test dask installed)
    from . import dask    # noqa (API import)
except ImportError:
    pass
