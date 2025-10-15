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
        log.warning(
            "Skipping nyc_taxi download: installed hvsampledata version has no 'nyc_taxi_remote'. "
            "Please upgrade hvsampledata when a new release is available."
        )
    else:
        try:
            nyc_taxi("pandas")
        except Exception as e:
            log.warning(f"hvsampledata.nyc_taxi_remote() failed: {e}")
