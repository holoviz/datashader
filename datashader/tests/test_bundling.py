import numpy as np
import pandas as pd

from datashader.bundling import directly_connect_edges, hammer_bundle

import pytest


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
                            'source': np.zeros(4),
                            'target': np.arange(1, 5)})
    edges_df.set_index('id')
    return edges_df


def assert_eq(a, b):
    assert a.equals(b)


def test_directly_connect(nodes, edges):
    # Expect four lines starting at center (0.5, 0.5) and terminating
    # at a different corner and NaN
    data = pd.DataFrame({'x': [0.5, 0.0, np.nan, 0.5, 1.0, np.nan,
                               0.5, 0.0, np.nan, 0.5, 1.0, np.nan],
                         'y': [0.5, 1.0, np.nan, 0.5, 1.0, np.nan,
                               0.5, 0.0, np.nan, 0.5, 0.0, np.nan]})
    expected = pd.DataFrame(data)

    given = directly_connect_edges(nodes, edges)
    assert_eq(given, expected)


def test_hammer_bundle(nodes, edges):
    # Expect four lines starting at center (0.5, 0.5) and terminating
    # with NaN
    data = pd.DataFrame({'x': [0.5, np.nan, 0.5, np.nan,
                               0.5, np.nan, 0.5, np.nan],
                         'y': [0.5, np.nan, 0.5, np.nan,
                               0.5, np.nan, 0.5, np.nan]})
    expected = pd.DataFrame(data)

    df = hammer_bundle(nodes, edges)

    starts = df[(df.x == 0.5) & (df.y == 0.5)]
    ends = df[df.isnull().any(axis=1)]
    given = pd.concat([starts, ends])
    given.sort_index(inplace=True)
    given.reset_index(drop=True, inplace=True)

    assert_eq(given, expected)
