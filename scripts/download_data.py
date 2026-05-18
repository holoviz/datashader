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
    import pooch

    retry(gds.get_path, "geoda.natregimes")
    # retry(gds.get_path, "nybb")
    retry(gds.get_path, "geoda health")

    # https://github.com/geopandas/geodatasets/issues/35
    _nybb = gds.data.query_name("nybb")
    _UA = "Mozilla/5.0 (X11; Linux x86_64; rv:150.0) Gecko/20100101 Firefox/150.0"
    _downloader = pooch.HTTPDownloader(headers={"User-Agent": _UA})
    retry(
        gds.api.CACHE.fetch,
        _nybb.filename,
        processor=pooch.Unzip(members=_nybb.members),
        downloader=_downloader,
    )



with suppress(ImportError):
    import hvsampledata as hvs

    retry(hvs.download, "nyc_taxi_remote")
