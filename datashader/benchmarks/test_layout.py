import pytest

import numpy as np
import pandas as pd

from datashader.layout import circular_layout, forceatlas2_layout, random_layout


@pytest.fixture
def nodes():
    # Four nodes arranged at the corners of a 200x200 square with one node
    # at the center
    nodes_df = pd.DataFrame({'id': np.arange(5),
                             'x': [0., -100., 100., -100., 100.],
                             'y': [0., 100., 100., -100., -100.]})
    return nodes_df.set_index('id')


@pytest.fixture
def edges():
    # Four edges originating from the center node and connected to each
    # corner
    edges_df = pd.DataFrame({'id': np.arange(4),
                             'source': np.zeros(4, dtype=np.int64),
                             'target': np.arange(1, 5)})
    return edges_df.set_index('id')


@pytest.mark.parametrize('layout', [random_layout, circular_layout, forceatlas2_layout])
@pytest.mark.benchmark(group="layout")
def test_layout(benchmark, nodes, edges, layout):
    benchmark(layout, nodes, edges)
