from pathlib import Path

import pyct.cmd

BASE_PATH = Path(__file__).resolve().parents[1]

try:
    import bokeh.sampledata
except ImportError:
    pass
else:
    bokeh.sampledata.download()

pyct.cmd.fetch_data(
    name="data",
    path=str(BASE_PATH / "examples"),
    datasets="datasets.yml",
)
