import pytest

import numpy as np
import pandas as pd

from datashader.layout import forceatlas2_layout


@pytest.fixture
def nodes():
    # Four nodes arranged at the corners of a 200x200 square with one node
    # at the center
    nodes_df = pd.DataFrame({'id': np.arange(5),
                             'x': [0., -100., 100., -100., 100.],
                             'y': [0., 100., 100., -100., -100.]})
    return nodes_df.set_index('id')


@pytest.fixture
def nodes_without_positions():
    nodes_df = pd.DataFrame({'id': np.arange(5)})
    return nodes_df.set_index('id')


@pytest.fixture
def edges():
    # Four edges originating from the center node and connected to each
    # corner
    edges_df = pd.DataFrame({'id': np.arange(4),
                             'source': np.zeros(4, dtype=np.int64),
                             'target': np.arange(1, 5)})
    return edges_df.set_index('id')


@pytest.fixture
def weighted_edges():
    # Four weighted edges originating from the center node and connected
    # to each corner
    edges_df = pd.DataFrame({'id': np.arange(4),
                             'source': np.zeros(4, dtype=np.int64),
                             'target': np.arange(1, 5),
                             'weight': np.ones(4)})
    return edges_df.set_index('id')


def test_forceatlas2_positioned_nodes_with_unweighted_edges(nodes, edges):
    df = forceatlas2_layout(nodes, edges)
    assert df.equals(nodes)


def test_forceatlas2_positioned_nodes_with_weighted_edges(nodes, weighted_edges):
    df = forceatlas2_layout(nodes, weighted_edges)
    assert df.equals(nodes)


def test_forceatlas2_unpositioned_nodes_with_unweighted_edges(nodes_without_positions, edges):
    df = forceatlas2_layout(nodes_without_positions, edges)
    assert len(nodes_without_positions) == len(df)
    assert not df.equals(nodes_without_positions)


def test_forceatlas2_unpositioned_nodes_with_weighted_edges(nodes_without_positions, weighted_edges):
    df = forceatlas2_layout(nodes_without_positions, weighted_edges)
    assert len(nodes_without_positions) == len(df)
    assert not df.equals(nodes_without_positions)
