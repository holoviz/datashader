# Datashader using AwkwardPandas vs SpatialPandas

import awkward as ak
import awkward_pandas as akpd
import colorcet as cc
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import spatialpandas.io as spdio
import time


directory = "/home/data/hv/nyc_buildings.parq"

cvs = ds.Canvas(plot_width=800, plot_height=600)

# SpatialPandas solution.
df = spdio.read_parquet(directory)

agg = cvs.polygons(source=df, geometry="geometry", agg=ds.count())
print("sp agg limits", agg.min().item(), agg.max().item())
im = tf.shade(agg, cmap=cc.bgy, how="eq_hist")
ds.utils.export_image(im, "ak_vs_sp0", background="black")

# AwkwardPandas solution.
array = ak.from_parquet(directory)
geometry = ak.fill_none(array.geometry, 999, axis=-1)
geometry = akpd.AwkwardExtensionArray(geometry)
hilbert_distance = akpd.AwkwardExtensionArray(array.hilbert_distance)
df = pd.DataFrame(data=dict(geometry=geometry, hilbert_distance=hilbert_distance))

agg = cvs.polygons(source=df, geometry="geometry", agg=ds.count())
print("ak agg limits", agg.min().item(), agg.max().item())

im = tf.shade(agg, cmap=cc.bgy, how="eq_hist")
ds.utils.export_image(im, "ak_vs_sp1", background="black")
