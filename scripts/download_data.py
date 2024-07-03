from contextlib import suppress

import pyct.cmd
from packaging.version import Version

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
    gds.get_path('geoda health')
