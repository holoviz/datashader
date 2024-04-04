import contextlib

with contextlib.suppress(Exception):
    # From Dask 2024.3.0 they now use `dask_expr` by default
    # https://github.com/dask/dask/issues/10995
    import dask

    dask.config.set(scheduler="synchronous")
    dask.config.set({"dataframe.query-planning": False})
