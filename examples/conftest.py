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
