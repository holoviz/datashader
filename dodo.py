import os
if "PYCTDEV_ECOSYSTEM" not in os.environ:
    os.environ["PYCTDEV_ECOSYSTEM"] = "conda"

from pyctdev import *  # noqa: api


def task_build_website():
    """
    Experimental: Build website
    Might migrate to pyctdev

    """
    return {
        'actions': [
            "datashader fetch-data --path=examples",
            "nbsite generate-rst --org pyviz --project-name datashader --skip '.*tiling.*'",
            "nbsite build --what=html --output=builtdocs",
        ]}

def task_release_website():
    """ Experimental: Release website """
    return {
        'actions': [
            "cd builtdocs",
            "aws s3 sync --delete --acl public-read . s3://datashader.org",
            "cd .."
        ]}