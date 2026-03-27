import sys
import time
from contextlib import suppress

from packaging.version import Version


def retry(func, *args, **kwargs):
    for i in range(5):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait = 10 * 2**i
            print(f"Attempt {i + 1} failed: {e}. Retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    return func(*args, **kwargs)


with suppress(ImportError):
    import pyct.cmd

    retry(pyct.cmd.fetch_data, name="data", path="examples", datasets="datasets.yml")


with suppress(ImportError):
    import bokeh

    # Replaced with bokeh_sampledata in 3.5
    if Version(bokeh.__version__).release < (3, 5, 0):
        import bokeh.sampledata

        retry(bokeh.sampledata.download)


with suppress(ImportError):
    import geodatasets as gds

    retry(gds.get_path, "geoda.natregimes")
    retry(gds.get_path, "nybb")
    retry(gds.get_path, "geoda health")


with suppress(ImportError):
    import hvsampledata as hvs

    retry(hvs.download, "nyc_taxi_remote")
