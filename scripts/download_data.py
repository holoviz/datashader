from contextlib import suppress

import pyct.cmd

pyct.cmd.fetch_data(name="data", path="examples", datasets="datasets.yml")


with suppress(ImportError):
    import bokeh.sampledata

    bokeh.sampledata.download()


with suppress(ImportError):
    import geodatasets as gds

    gds.get_path("geoda.natregimes")
    gds.get_path("nybb")
