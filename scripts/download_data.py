from contextlib import suppress
from packaging.version import Version

with suppress(ImportError):
    import pyct.cmd
    from datashader import _warn_pyct_deprecated

    _warn_pyct_deprecated(stacklevel=1)
    pyct.cmd.fetch_data(name="data", path="examples", datasets="datasets.yml")


with suppress(ImportError):
    import bokeh

    # Replaced with bokeh_sampledata in 3.5
    if Version(bokeh.__version__) < Version("3.5"):
        import bokeh.sampledata

        bokeh.sampledata.download()


with suppress(ImportError):
    import geodatasets as gds

    gds.get_path("geoda.natregimes")
    gds.get_path("nybb")
    gds.get_path("geoda health")


with suppress(ImportError):
    import hvsampledata as hvs

    path = hvs.download("nyc_taxi_remote")
