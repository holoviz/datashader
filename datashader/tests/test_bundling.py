import numpy as np
import pandas as pd

from datashader.bundling import directly_connect_edges, hammer_bundle


nodes_df = pd.DataFrame({'id': np.arange(5),
                         'x': [0, -100, 100, -100, 100],
                         'y': [0, 100, 100, -100, -100]})
nodes_df.set_index('id')

edges_df = pd.DataFrame({'id': np.arange(4),
                         'source': np.zeros(4),
                         'target': np.arange(1, 5)})
edges_df.set_index('id')


def assert_eq(a, b):
    assert a.equals(b)


def test_directly_connect():
    data = pd.DataFrame({'x': [0.5, 0.0, np.nan, 0.5, 1.0, np.nan,
                               0.5, 0.0, np.nan, 0.5, 1.0, np.nan],
                         'y': [0.5, 1.0, np.nan, 0.5, 1.0, np.nan,
                               0.5, 0.0, np.nan, 0.5, 0.0, np.nan]})
    expected = pd.DataFrame(data)

    given = directly_connect_edges(nodes_df, edges_df)
    assert_eq(given, expected)


def test_hammer_bundle():
    data = pd.DataFrame({'x': [0.5, np.nan, 0.5, np.nan,
                               0.5, np.nan, 0.5, np.nan],
                         'y': [0.5, np.nan, 0.5, np.nan,
                               0.5, np.nan, 0.5, np.nan]})
    expected = pd.DataFrame(data)

    df = hammer_bundle(nodes_df, edges_df)

    starts = df[(df.x == 0.5) & (df.y == 0.5)]
    ends = df[df.isnull().any(axis=1)]
    given = pd.concat([starts, ends])
    given.sort_index(inplace=True)
    given.reset_index(drop=True, inplace=True)

    assert_eq(given, expected)
