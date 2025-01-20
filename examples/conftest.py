from importlib.util import find_spec
from packaging.version import Version

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

if find_spec("dask") is not None:
    import dask

    # Spatialpandas does not support dask-expr, which is
    # only available from this version.
    if Version(dask.__version__).release >= (2025, 1, 0):
        collect_ignore_glob += [
            "user_guide/8_Polygons.ipynb",
        ]
