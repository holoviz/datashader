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


with suppress(ImportError):
    import hvsampledata as hvs

    # Temp workaround for CI until a new hvsampledata release is available.
    import logging
    import sys

    log = logging.getLogger("datashader.download_data")
    if not log.handlers:
        # Ensure at least one handler so warnings are visible in CI logs
        handler = logging.StreamHandler(stream=sys.stderr)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        log.addHandler(handler)

    nyc_taxi = getattr(hvs, "nyc_taxi_remote", None)
    if nyc_taxi is None:
        # Try a direct download of the parquet file as a temporary fallback.
        URL = "https://datasets.holoviz.org/nyc_taxi/v2/nyc_taxi_wide.parq"

        from pathlib import Path
        dest = Path(hvs._DATAPATH) / "nyc_taxi_wide.parq"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            import requests

            tmp = dest.with_suffix(".part")
            with requests.get(URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=10_485_760):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dest)
        except Exception:
            # Fallback if requests not available or streaming fails
            import urllib.request

            urllib.request.urlretrieve(URL, dest)
        log.info("nyc_taxi parquet downloaded to %s", dest)
    else:
        try:
            nyc_taxi("pandas")
        except Exception as e:
            log.warning(f"hvsampledata.nyc_taxi_remote() failed: {e}")
