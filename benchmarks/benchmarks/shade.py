import colorcet
import datashader.transfer_functions as tf
import numpy as np
import xarray as xr


class Shade:
    param_names = ("canvas_size", "how")
    params = ([300, 1000], ["linear", "log", "eq_hist"])

    def setup(self, canvas_size, how):
        rng = np.random.default_rng(349120)
        data = rng.random((canvas_size, canvas_size), dtype=np.float32)
        data[data < 0.1] = np.nan  # Want some nans in data.
        x = np.arange(canvas_size, dtype=np.float32)
        y = np.arange(canvas_size, dtype=np.float32)
        self.agg = xr.DataArray(data=data, dims=["y", "x"], coords=dict(x=x, y=y))

    def time_shade(self, canvas_size, how):
        tf.shade(self.agg, how=how, cmap=colorcet.fire)


class ShadeCategorical:
    param_names = ("canvas_size", "how", "category_count")
    params = ([300, 1000], ["linear", "log", "eq_hist"], [3, 10, 30, 100])

    def setup(self, canvas_size, how, category_count):
        rng = np.random.default_rng(349120)
        data = rng.random((canvas_size, canvas_size, category_count), dtype=np.float32)
        data[data < 0.1] = np.nan  # Want some nans in data.
        x = np.arange(canvas_size, dtype=np.float32)
        y = np.arange(canvas_size, dtype=np.float32)
        cat = [f"cat{i}" for i in range(category_count)]
        self.agg = xr.DataArray(data=data, dims=["y", "x", "cat"], coords=dict(x=x, y=y, cat=cat))

        random_colors = rng.choice(colorcet.rainbow, category_count)
        self.color_key = {k: v for k, v in zip(cat, random_colors)}

    def time_shade_categorical(self, canvas_size, how, category_count):
        tf.shade(self.agg, how=how, color_key=self.color_key)
