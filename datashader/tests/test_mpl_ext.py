from __future__ import annotations
import pytest

pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import datashader as ds
from datashader.mpl_ext import dsshow


df = pd.DataFrame(
    {
        "x": np.array(([0.0] * 10 + [1] * 10)),
        "y": np.array(([0.0] * 5 + [1] * 5 + [0] * 5 + [1] * 5)),
    }
)


def test_image_initialize():
    plt.figure(dpi=100)
    ax = plt.subplot(111)
    da = dsshow(df, ds.Point("x", "y"), ax=ax)

    data = da.get_ds_data()
    assert data[0, 0] == 5
    assert data[0, -1] == 5
    assert data[-1, 0] == 5
    assert data[-1, -1] == 5


def test_image_update():
    fig = plt.figure(dpi=100)
    ax = plt.subplot(111)
    da = dsshow(df, ds.Point("x", "y"), ax=ax)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    fig.canvas.draw()

    data = da.get_ds_data()
    assert data[0, 0] == 5
    assert data[0, -1] == 0
    assert data[-1, 0] == 0
    assert data[-1, -1] == 0
