import pytest

import numpy as np
import pandas as pd

from datashader.bundling import directly_connect_edges, hammer_bundle
from datashader.layout import circular_layout, forceatlas2_layout, random_layout

skimage = pytest.importorskip("skimage")


@pytest.fixture
def nodes():
    # Four nodes arranged at the corners of a 200x200 square with one node
    # at the center
    nodes_df = pd.DataFrame({'id': np.arange(5),
                             'x': [0.0, -100.0, 100.0, -100.0, 100.0],
                             'y': [0.0, 100.0, 100.0, -100.0, -100.0]})
    nodes_df.set_index('id')
    return nodes_df


@pytest.fixture
def edges():
    # Four edges originating from the center node and connected to each
    # corner
    edges_df = pd.DataFrame({'id': np.arange(4),
                             'source': np.zeros(4, dtype=int),
                             'target': np.arange(1, 5)})
    edges_df.set_index('id')
    return edges_df


@pytest.mark.parametrize('bundle', [directly_connect_edges, hammer_bundle])
@pytest.mark.parametrize('layout', [random_layout, circular_layout, forceatlas2_layout])
@pytest.mark.benchmark(group="bundling")
def test_bundle(benchmark, nodes, edges, layout, bundle):
    node_positions = layout(nodes, edges)
    benchmark(bundle, node_positions, edges)
