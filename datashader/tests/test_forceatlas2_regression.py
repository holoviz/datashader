from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datashader.layout import forceatlas2_layout
from pandas.testing import assert_frame_equal


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("linlog", [False, True])
def test_forceatlas2_two_node_update_and_frame_contract(dtype, weighted, linlog):
    nodes = pd.DataFrame(
        {"x": np.array([0.0, 1.0], dtype=dtype),
         "y": np.array([0.0, 0.0], dtype=dtype),
         "label": ["left", "right"]},
        index=pd.Index(["left", "right"], name="node"),
    )
    edges = pd.DataFrame({"source": ["left"], "target": ["right"], "weight": [2.0]})
    nodes_before = nodes.copy(deep=True)
    edges_before = edges.copy(deep=True)

    kwargs = {"iterations": 1}
    if linlog:
        kwargs["linlog"] = True
    if weighted:
        kwargs["weight"] = "weight"
    result = forceatlas2_layout(nodes, edges, **kwargs)

    assert_frame_equal(nodes, nodes_before)
    assert_frame_equal(edges, edges_before)
    assert result.index.equals(nodes.index)
    assert result.columns.tolist() == nodes.columns.tolist()
    assert result["label"].tolist() == nodes["label"].tolist()
    np.testing.assert_allclose(
        result[["x", "y"]].to_numpy(),
        np.array([[0.1, 0.0], [0.9, 0.0]]),
        rtol=0,
        atol=1e-6,
    )


def test_forceatlas2_seeded_unpositioned_custom_id_is_repeatable():
    nodes = pd.DataFrame(
        {"name": ["left", "right"], "label": [1, 2]},
        index=pd.Index([20, 10], name="row"),
    )
    edges = pd.DataFrame({"source": ["left"], "target": ["right"]})
    nodes_before = nodes.copy(deep=True)
    edges_before = edges.copy(deep=True)

    first = forceatlas2_layout(nodes, edges, id="name", seed=7, iterations=2)
    second = forceatlas2_layout(nodes, edges, id="name", seed=7, iterations=2)

    assert_frame_equal(first, second)
    assert_frame_equal(nodes, nodes_before)
    assert_frame_equal(edges, edges_before)
    assert first.index.equals(nodes.index)
    assert first["name"].tolist() == nodes["name"].tolist()
    assert np.isfinite(first[["x", "y"]].to_numpy()).all()
