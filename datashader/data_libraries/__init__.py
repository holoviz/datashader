from importlib.util import find_spec
from warnings import warn

from . import pandas, xarray


if find_spec("dask"):
    if not find_spec("pyarrow"):
        warn(
            "dask requires pyarrow to work with datashader.",
            RuntimeWarning,
            stacklevel=3,
        )
    else:
        from . import dask

if find_spec("cudf") and find_spec("cupy"):
    from . import cudf

    if find_spec("dask_cudf"):
        from . import dask_cudf


__all__ = (
    "pandas",
    "xarray",
    "dask",
    "cudf",
    "dask_cudf",
)
