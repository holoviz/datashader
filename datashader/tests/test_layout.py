from __future__ import annotations
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


@pytest.mark.parametrize('bundle', [directly_connect_edges, hammer_bundle])
@pytest.mark.parametrize('layout', [random_layout, circular_layout, forceatlas2_layout])
def test_renamed_columns(nodes, weighted_edges, bundle, layout):
    nodes = nodes.rename(columns={'x': 'xx', 'y': 'yy'})
    edges = weighted_edges.rename(columns={'source': 'src', 'target': 'dst', 'weight': 'w'})

    node_positions = layout(nodes, edges, x='xx', y='yy', source='src', target='dst', weight='w')
    df = bundle(node_positions, edges, x='xx', y='yy', source='src', target='dst', weight='w')

    assert 'xx' in df and 'x' not in df
    assert 'yy' in df and 'y' not in df
    assert 'w' in df and 'weight' not in df


def test_forceatlas2_positioned_nodes_with_unweighted_edges(nodes, edges):
    df = forceatlas2_layout(nodes, edges)
    assert len(nodes) == len(df)
    assert not df.equals(nodes)


def test_forceatlas2_positioned_nodes_with_weighted_edges(nodes, weighted_edges):
    df = forceatlas2_layout(nodes, weighted_edges)
    assert len(nodes) == len(df)
    assert not df.equals(nodes)


def test_forceatlas2_unpositioned_nodes_with_unweighted_edges(nodes_without_positions, edges):
    df = forceatlas2_layout(nodes_without_positions, edges)
    assert len(nodes_without_positions) == len(df)
    assert not df.equals(nodes_without_positions)


def test_forceatlas2_unpositioned_nodes_with_weighted_edges(nodes_without_positions,
                                                            weighted_edges):
    df = forceatlas2_layout(nodes_without_positions, weighted_edges)
    assert len(nodes_without_positions) == len(df)
    assert not df.equals(nodes_without_positions)


def test_random_layout(nodes_without_positions, edges):
    expected_x = [0.511821624, 0.144159612, 0.311831452, 0.827702593, 0.549593687]
    expected_y = [0.950463696, 0.948649447, 0.423326448, 0.409199136, 0.027559113]
    df = random_layout(nodes_without_positions, edges, seed=1)

    assert np.allclose(df['x'], expected_x)
    assert np.allclose(df['y'], expected_y)


def test_uniform_circular_layout(nodes_without_positions, edges):
    expected_x = [1.0, 0.654508497187, 0.095491502813, 0.095491502813, 0.654508497187]
    expected_y = [0.5, 0.975528258148, 0.793892626146, 0.206107373854, 0.024471741852]

    df = circular_layout(nodes_without_positions, edges)

    assert np.allclose(df['x'], expected_x)
    assert np.allclose(df['y'], expected_y)


def test_random_circular_layout(nodes_without_positions, edges):
    expected_x = [0.00137865, 0.97597639, 0.80853537, 0.97420002, 0.31060039]
    expected_y = [0.46289541, 0.34687760, 0.89345383, 0.34146188, 0.96273944]
    df = circular_layout(nodes_without_positions, edges, uniform=False, seed=1)

    assert np.allclose(df['x'], expected_x)
    assert np.allclose(df['y'], expected_y)
