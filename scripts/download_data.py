from contextlib import suppress
from packaging.version import Version

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
