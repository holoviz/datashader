"""Assign coordinates to the nodes of a graph.

Timothee Poisot's `nxfa2` is the original implementation of the main
algorithm.

.. _nxfa2:
   https://github.com/tpoisot/nxfa2
"""

from __future__ import absolute_import, division, print_function

import numba as nb
import numpy as np
import param
import scipy as sp


def _extract_points_from_nodes(nodes):
    if 'x' in nodes.columns and 'y' in nodes.columns:
        points = np.asarray(nodes[['x', 'y']])
    else:
        points = np.asarray(np.random.random((len(nodes), 2)))
    return points


def _convert_edges_to_sparse_matrix(edges):
    if 'weight' in edges:
        weights = edges['weight']
    else:
        weights = np.ones(len(edges))

    A = sp.sparse.coo_matrix((weights, (edges['source'], edges['target'])))
    return A.tolil()


def _merge_points_with_nodes(nodes, points):
    n = nodes.copy()
    n['x'] = points[:, 0]
    n['y'] = points[:, 1]
    return n


@nb.jit(nogil=True)
def cooling(matrix, points, temperature, params):
    dt = temperature / float(params.iterations + 1)
    displacement = np.zeros((params.dim, len(points)))
    for iteration in range(params.iterations):
        displacement *= 0
        for i in range(matrix.shape[0]):
            # difference between this row's node position and all others
            delta = (points[i] - points).T

            # distance between points
            distance = np.sqrt((delta ** 2).sum(axis=0))

            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)

            # the adjacency matrix row
            ai = np.asarray(matrix.getrowview(i).toarray())

            # displacement "force"
            dist = params.k * params.k / distance ** 2

            if params.nohubs:
                dist = dist / float(ai.sum(axis=1) + 1)
            if params.linlog:
                dist = np.log(dist + 1)
            displacement[:, i] += (delta * (dist - ai * distance / params.k)).sum(axis=1)

        # update points
        length = np.sqrt((displacement ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.01, length)
        points += (displacement * temperature / length).T

        # cool temperature
        temperature -= dt


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
        points = _extract_points_from_nodes(nodes)
        matrix = _convert_edges_to_sparse_matrix(edges)

        if p.k is None:
            p.k = np.sqrt(1.0 / len(points))

        # the initial "temperature" is about .1 of domain area (=1x1)
        # this is the largest step allowed in the dynamics.
        temperature = 0.1

        # simple cooling scheme.
        # linearly step down by dt on each iteration so last iteration is size dt.
        cooling(matrix, points, temperature, p)

        # Return the nodes with updated positions
        return _merge_points_with_nodes(nodes, points)
