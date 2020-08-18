from __future__ import absolute_import
import pytest
pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import datashader as ds
from datashader.mpl_ext import dsshow

axis = ds.core.LinearAxis()
lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)
coords = [lincoords, lincoords]
dims = ['y', 'x']

df = pd.DataFrame({'x': np.array(([0.] * 10 + [1] * 10)),
                   'y': np.array(([0.] * 5 + [1] * 5 + [0] * 5 + [1] * 5))})


def test_image_initialize():
    plt.figure(dpi=100)
    ax = plt.subplot(111)
    da = dsshow(
        df,
        ds.Point('x', 'y'),
        ax=ax,
        height_scale=1,
        width_scale=1,
    )

    data = da.get_ds_data()
    assert data[0, 0] == 5
    assert data[0, -1] == 5
    assert data[-1, 0] == 5
    assert data[-1, -1] == 5

    img = da.get_ds_image()
    rgba = da.get_array()
    out  = rgba.view(dtype=np.uint32).reshape(rgba.shape[:-1])
    assert np.array_equal(img, out)


def test_image_update():
    fig = plt.figure(dpi=100)
    ax = plt.subplot(111)
    da = dsshow(
        df,
        ds.Point('x', 'y'),
        ax=ax,
        height_scale=1,
        width_scale=1,
    )
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    fig.canvas.draw()

    data = da.get_ds_data()
    assert data[0, 0] == 5
    assert data[0, -1] == 0
    assert data[-1, 0] == 0
    assert data[-1, -1] == 0

    img = da.get_ds_image()
    rgba = da.get_array()
    out  = rgba.view(dtype=np.uint32).reshape(rgba.shape[:-1])
    assert np.array_equal(img, out)
