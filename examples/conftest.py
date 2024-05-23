from importlib.util import find_spec

import platform

collect_ignore_glob = []

if find_spec("geopandas") is None:
    collect_ignore_glob += [
        "user_guide/13_Geopandas.ipynb",
    ]

if find_spec("spatialpandas") is None:
    collect_ignore_glob += [
        "user_guide/7_Networks.ipynb",
        "user_guide/8_Polygons.ipynb",
    ]

# 2023-07-21 with following error:
# nbclient.exceptions.CellTimeoutError: A cell timed out while it was being executed, after 300 seconds.
# Here is a preview of the cell contents:
# -------------------
# import holoviews.operation.datashader as hd
# import holoviews as hv
# hv.extension("bokeh")
# circle = hv.Graph(edges, label='Bokeh edges').opts(node_size=5)
# hnodes = circle.nodes.opts(size=5)
# dscirc = (hd.spread(hd.datashade(circle))*hnodes).relabel("Datashader edges")
# circle + dscirc
# ------------------------------ Captured log call ------------------------------
# ERROR    traitlets:client.py:841 Timeout waiting for execute reply (300s).
if platform.system() == "Windows":
    collect_ignore_glob += [
        "user_guide/7_Networks.ipynb",
    ]
