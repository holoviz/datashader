from __future__ import annotations
import pytest
skimage = pytest.importorskip("skimage")

import numpy as np
import pandas as pd

from datashader.bundling import directly_connect_edges, hammer_bundle
from datashader.layout import circular_layout, forceatlas2_layout, random_layout


@pytest.fixture
def nodes():
    # Four nodes arranged at the corners of a 200x200 square with one node
    # at the center
    nodes_df = pd.DataFrame({'id': np.arange(5),
                             'x': [0, -100, 100, -100, 100],
                             'y': [0, 100, 100, -100, -100]})
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


@pytest.fixture
def weighted_edges():
    # Four weighted edges originating from the center node and connected
    # to each corner
    edges_df = pd.DataFrame({'id': np.arange(4),
                             'source': np.zeros(4, dtype=int),
                             'target': np.arange(1, 5),
                             'weight': np.ones(4)})
    edges_df.set_index('id')
    return edges_df


def test_immutable_nodes(nodes, edges):
    # Expect nodes to remain immutable after any bundling operation
    original = nodes.copy()
    directly_connect_edges(nodes, edges)
    assert original.equals(nodes)


@pytest.mark.parametrize('bundle', [directly_connect_edges, hammer_bundle])
def test_renamed_columns(nodes, weighted_edges, bundle):
    nodes = nodes.rename(columns={'x': 'xx', 'y': 'yy'})
    edges = weighted_edges.rename(columns={'source': 'src', 'target': 'dst', 'weight': 'w'})

    df = bundle(nodes, edges, x='xx', y='yy', source='src', target='dst', weight='w')

    assert 'xx' in df and 'x' not in df
    assert 'yy' in df and 'y' not in df
    assert 'w' in df and 'weight' not in df


@pytest.mark.parametrize('bundle', [directly_connect_edges, hammer_bundle])
@pytest.mark.parametrize('layout', [random_layout, circular_layout, forceatlas2_layout])
def test_same_path_endpoints(layout, bundle):
    # Expect path endpoints to match original edge source/target
    edges = pd.DataFrame({'id': [0], 'source': [0], 'target': [1]}).set_index('id')
    nodes = pd.DataFrame({'id': np.unique(edges.values)}).set_index('id')

    node_positions = layout(nodes, edges)
    bundled = bundle(node_positions, edges)

    source, target = edges.iloc[0]
    expected_source = node_positions.loc[source]
    expected_target = node_positions.loc[target]

    actual_source = bundled.loc[0]
    actual_target = bundled.loc[len(bundled)-2]

    assert np.allclose(expected_source, actual_source)
    assert np.allclose(expected_target, actual_target)


@pytest.mark.parametrize("include_edge_id", [True, False])
def test_directly_connect_with_weights(nodes, weighted_edges, include_edge_id):
    # Expect four lines starting at center (0.5, 0.5) and terminating
    # at a different corner and NaN
    data = pd.DataFrame({'edge_id':
                            [1.0, 1.0, np.nan, 2.0, 2.0, np.nan,
                             3.0, 3.0, np.nan, 4.0, 4.0, np.nan],
                         'x':
                            [0.0, -100.0, np.nan, 0.0, 100.0, np.nan,
                             0.0, -100.0, np.nan, 0.0, 100.0, np.nan],
                         'y':
                            [0.0, 100.0, np.nan, 0.0, 100.0, np.nan,
                             0.0, -100.0, np.nan, 0.0, -100.0, np.nan]})
    columns = ['edge_id', 'x', 'y'] if include_edge_id else ['x', 'y']
    expected = pd.DataFrame(data, columns=columns)

    given = directly_connect_edges(nodes, weighted_edges, include_edge_id=include_edge_id)
    assert given.equals(expected)


@pytest.mark.parametrize("include_edge_id", [True, False])
def test_directly_connect_without_weights(nodes, edges, include_edge_id):
    # Expect four lines starting at center (0.5, 0.5) and terminating
    # at a different corner and NaN
    data = pd.DataFrame({'edge_id':
                            [1.0, 1.0, np.nan, 2.0, 2.0, np.nan,
                             3.0, 3.0, np.nan, 4.0, 4.0, np.nan],
                         'x':
                            [0.0, -100.0, np.nan, 0.0, 100.0, np.nan,
                             0.0, -100.0, np.nan, 0.0, 100.0, np.nan],
                         'y':
                            [0.0, 100.0, np.nan, 0.0, 100.0, np.nan,
                             0.0, -100.0, np.nan, 0.0, -100.0, np.nan]})
    columns = ['edge_id', 'x', 'y'] if include_edge_id else ['x', 'y']
    expected = pd.DataFrame(data, columns=columns)

    given = directly_connect_edges(nodes, edges, include_edge_id=include_edge_id)
    assert given.equals(expected)


@pytest.mark.parametrize("include_edge_id", [True, False])
def test_hammer_bundle_with_weights(nodes, weighted_edges, include_edge_id):
    # Expect four lines starting at center (0.0, 0.0) and terminating
    # with NaN
    data = pd.DataFrame({'edge_id':
                            [1.0, np.nan, 2.0, np.nan,
                             3.0, np.nan, 4.0, np.nan],
                         'x':
                            [0.0, np.nan, 0.0, np.nan,
                             0.0, np.nan, 0.0, np.nan],
                         'y':
                            [0.0, np.nan, 0.0, np.nan,
                             0.0, np.nan, 0.0, np.nan],
                         'weight':
                            [1.0, np.nan, 1.0, np.nan,
                             1.0, np.nan, 1.0, np.nan]})
    columns = ['edge_id', 'x', 'y', 'weight'] if include_edge_id else ['x', 'y', 'weight']
    expected = pd.DataFrame(data, columns=columns)

    df = hammer_bundle(nodes, weighted_edges, include_edge_id=include_edge_id)

    starts = df[(df.x == 0.0) & (df.y == 0.0)]
    ends = df[df.isnull().any(axis=1)]
    given = pd.concat([starts, ends])
    given.sort_index(inplace=True)
    given.reset_index(drop=True, inplace=True)

    assert given.equals(expected)


@pytest.mark.parametrize("include_edge_id", [True, False])
def test_hammer_bundle_without_weights(nodes, edges, include_edge_id):
    # Expect four lines starting at center (0.0, 0.0) and terminating
    # with NaN
    data = pd.DataFrame({'edge_id':
                            [1.0, np.nan, 2.0, np.nan,
                             3.0, np.nan, 4.0, np.nan],
                         'x':
                            [0.0, np.nan, 0.0, np.nan,
                             0.0, np.nan, 0.0, np.nan],
                         'y':
                            [0.0, np.nan, 0.0, np.nan,
                             0.0, np.nan, 0.0, np.nan]})
    columns = ['edge_id', 'x', 'y'] if include_edge_id else ['x', 'y']
    expected = pd.DataFrame(data, columns=columns)

    df = hammer_bundle(nodes, edges, include_edge_id=include_edge_id)

    starts = df[(df.x == 0.0) & (df.y == 0.0)]
    ends = df[df.isnull().any(axis=1)]
    given = pd.concat([starts, ends])
    given.sort_index(inplace=True)
    given.reset_index(drop=True, inplace=True)

    assert given.equals(expected)
