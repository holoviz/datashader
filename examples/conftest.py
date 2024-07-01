import os
import platform
from importlib.util import find_spec

collect_ignore_glob = [
    "tiling.ipynb",
]

if find_spec("geopandas") is None:
    collect_ignore_glob += [
        "user_guide/13_Geopandas.ipynb",
    ]

if find_spec("spatialpandas") is None:
    collect_ignore_glob += [
        "user_guide/7_Networks.ipynb",
        "user_guide/8_Polygons.ipynb",
    ]


# Will give an exit code 143 on Github Actions
# First seen in: https://github.com/holoviz/datashader/pull/1347
# Which tried to update to Geopandas 1.0, but during development
# Numpy 2 was also able to be solved for on Conda-Forge
if platform.system() == "Linux" and os.environ.get("GITHUB_ACTIONS"):
    collect_ignore_glob += [
        "user_guide/8_Polygons.ipynb",
    ]
