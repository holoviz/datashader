from pathlib import Path
from contextlib import suppress

import pyct.cmd

BASE_PATH = Path(__file__).resolve().parents[1]


pyct.cmd.fetch_data(name="data", path=str(BASE_PATH / "examples"), datasets="datasets.yml")


with suppress(ImportError):
    import bokeh.sampledata

    bokeh.sampledata.download()
