"""Assign coordinates to the nodes of a graph.

Timothee Poisot's `nxfa2` is the original implementation of the main
algorithm.

.. _nxfa2:
   https://github.com/tpoisot/nxfa2
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import param
import scipy as sp


def _convert_graph_with_positions_to_dataframes(graph, pos):
    """
    Convert NetworkX graph with associated positions into two dataframes.

    In a NetworkX graph, each edge can have its own independent attributes. One
    edge can have a different set of attributes than another edge. This means
    we have to assign a default weight value when converting to dataframes.
    """
    nodes = pd.DataFrame()
    for node, xy in zip(graph, pos):
        nodes = nodes.append({'id': node, 'x': xy[0], 'y': xy[1]}, ignore_index=True)

    nodes['id'].astype(np.int32)
    nodes = nodes.set_index('id')

    edges = pd.DataFrame()
    for edge in graph.edges():
        edge_attributes = graph[edge[0]][edge[1]]
        if 'weight' in edge_attributes:
            weight = edge_attributes['weight']
        else:
            weight = 1
        edges = edges.append({'source': edge[0], 'target': edge[1], 'weight': weight}, ignore_index=True)

    edges['source'].astype(np.int32)
    edges['target'].astype(np.int32)

    return nodes, edges


class forceatlas2_layout(param.ParameterizedFunction):
    """
    Assign coordinates to the nodes of a graph.

    This is a force-directed graph layout algorithm.

    .. _ForceAtlas2:
       http://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0098679&type=printable
    """

    def __call__(self, graph, iterations=10, linlog=False, pos=None, nohubs=False, k=None, dim=2):
        """
        Parameters
        ----------
        graph : networkx.Graph
            The NetworkX graph to layout
        iterations : int
            Number of iterations
        linlog : bool
            Whether to use logarithmic attraction force
        pos : ndarray
            Initial positions for the given nodes
        nohubs : bool
            Whether to grant authorities (nodes with a high indegree) a
            more central position than hubs (nodes with a high outdegree)
        k : float
            Compensates for the repulsion for nodes that are far away
            from the center. Defaults to the inverse of the number of
            nodes.
        dim : int
            Coordinate dimensions of each node.

        Returns
        -------
        nodes, edges : pandas.DataFrame
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError('install networkx to use this feature')

        # This comes from the sparse FR layout in NetworkX
        A = nx.to_scipy_sparse_matrix(graph, dtype='f')
        nnodes, _ = A.shape

        try:
            A = A.tolil()
        except Exception:
            A = (sp.sparse.coo_matrix(A)).tolil()
        if pos is None:
            pos = np.asarray(np.random.random((nnodes, dim)), dtype=A.dtype)
        else:
            pos = pos.astype(A.dtype)
        if k is None:
            k = np.sqrt(1.0 / nnodes)

        # the initial "temperature" is about .1 of domain area (=1x1)
        # this is the largest step allowed in the dynamics.
        t = 0.1

        # simple cooling scheme.
        # linearly step down by dt on each iteration so last iteration is size dt.
        dt = t / float(iterations + 1)
        displacement = np.zeros((dim, nnodes))
        for iteration in range(iterations):
            displacement *= 0
            for i in range(A.shape[0]):
                # difference between this row's node position and all others
                delta = (pos[i] - pos).T

                # distance between points
                distance = np.sqrt((delta ** 2).sum(axis=0))

                # enforce minimum distance of 0.01
                distance = np.where(distance < 0.01, 0.01, distance)

                # the adjacency matrix row
                ai = np.asarray(A.getrowview(i).toarray())

                # displacement "force"
                dist = k * k / distance ** 2

                if nohubs:
                    dist = dist / float(ai.sum(axis=1) + 1)
                if linlog:
                    dist = np.log(dist + 1)
                displacement[:, i] += (delta * (dist - ai * distance / k)).sum(axis=1)

            # update positions
            length = np.sqrt((displacement ** 2).sum(axis=0))
            length = np.where(length < 0.01, 0.01, length)
            pos += (displacement * t / length).T

            # cool temperature
            t -= dt

        # Return the layout
        return _convert_graph_with_positions_to_dataframes(graph, pos)
