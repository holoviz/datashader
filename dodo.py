import os
if "PYCTDEV_ECOSYSTEM" not in os.environ:
    os.environ["PYCTDEV_ECOSYSTEM"] = "conda"

from pyctdev import *  # noqa: api

def task_pip_on_conda():
    """Experimental: provide pip build env via conda"""
    return {'actions':[
        # some ecosystem=pip build tools must be installed with conda when using conda...
        'conda install -y pip twine wheel rfc3986 keyring',
        # ..and some are only available via conda-forge
        'conda install -y tox virtualenv',
    ]}
