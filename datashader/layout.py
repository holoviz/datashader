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


def _extract_points_from_nodes(nodes):
    if 'x' in nodes.columns and 'y' in nodes.columns:
        points = np.asarray(nodes[['x', 'y']])
    else:
        points = np.asarray(np.random.random((len(nodes), 2)))
    return points


def _convert_edges_to_sparse_matrix(edges):
    nedges = len(edges)

    if 'weight' in edges:
        weights = edges['weights']
    else:
        weights = np.ones(nedges)

    A = sp.sparse.coo_matrix((weights, (edges['source'], edges['target'])), shape=(nedges, nedges))
    return A.tolil()


def _merge_points_with_nodes(nodes, points):
    nodes['x'] = points[:, 0]
    nodes['y'] = points[:, 1]
    return nodes


class forceatlas2_layout(param.ParameterizedFunction):
    """
    Assign coordinates to the nodes of a graph.

    This is a force-directed graph layout algorithm.

    .. _ForceAtlas2:
       http://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0098679&type=printable
    """

    iterations = param.Integer(default=10, bounds=(1, None), doc="""
        Number of passes for the layout algorithm""")

    linlog = param.Boolean(False, doc="""
        Whether to use logarithmic attraction force""")

    nohubs = param.Boolean(False, doc="""
        Whether to grant authorities (nodes with a high indegree) a
        more central position than hubs (nodes with a high outdegree)""")

    k = param.Number(default=None, doc="""
        Compensates for the repulsion for nodes that are far away
        from the center. Defaults to the inverse of the number of
        nodes.""")

    dim = param.Integer(default=2, bounds=(1, None), doc="""
        Coordinate dimensions of each node""")

    def __call__(self, nodes, edges, **params):
        p = param.ParamOverrides(self, params)

        # Convert graph into sparse adjacency matrix and array of points
        nnodes = len(nodes)
        points = _extract_points_from_nodes(nodes)
        A = _convert_edges_to_sparse_matrix(edges)

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
                delta = (points[i] - points).T

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

            # update points
            length = np.sqrt((displacement ** 2).sum(axis=0))
            length = np.where(length < 0.01, 0.01, length)
            points += (displacement * t / length).T

            # cool temperature
            t -= dt

        # Return the nodes with updated positions
        return _merge_points_with_nodes(nodes, points)
