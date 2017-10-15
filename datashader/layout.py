"""Assign coordinates to the nodes of a graph.
"""

from __future__ import absolute_import, division, print_function

import numba as nb
import numpy as np
import param
import scipy.sparse


class LayoutAlgorithm(param.ParameterizedFunction):
    """
    Baseclass for all graph layout algorithms.
    """

    __abstract = True

    seed = param.Integer(default=None, bounds=(0, 2**32-1), doc="""
        Random seed used to initialize the pseudo-random number
        generator.""")

    x = param.String(default='x', doc="""
        Column name for each node's x coordinate.""")

    y = param.String(default='y', doc="""
        Column name for each node's y coordinate.""")

    source = param.String(default='source', doc="""
        Column name for each edge's source.""")

    target = param.String(default='target', doc="""
        Column name for each edge's target.""")

    weight = param.String(default='weight', allow_None=True, doc="""
        Column name for each edge weight. If None, weights are ignored.""")

    def __call__(self, nodes, edges, **params):
        """
        This method takes two dataframes representing a graph's nodes
        and edges respectively. For the nodes dataframe, the only
        column is id. For the edges dataframe, the columns are id,
        source, target, and (optionally) weight.

        Each layout algorithm will use the two dataframes as appropriate to
        assign positions to the nodes. Upon generating positions, this
        method will return a copy of the original nodes dataframe with
        two additional columns for the x and y coordinates.
        """
        return NotImplementedError


class random_layout(LayoutAlgorithm):
    """
    Assign coordinates to the nodes randomly.
    """

    def __call__(self, nodes, edges, **params):
        p = param.ParamOverrides(self, params)

        np.random.seed(p.seed)

        df = nodes.copy()
        points = np.asarray(np.random.random((len(df), 2)))

        df[p.x] = points[:, 0]
        df[p.y] = points[:, 1]

        return df


class circular_layout(LayoutAlgorithm):
    """
    Assign coordinates to the nodes along a circle.

    The points on the circle can be spaced either uniformly or randomly.
    """

    uniform = param.Boolean(True, doc="""
        Whether to distribute nodes evenly on circle""")

    def __call__(self, nodes, edges, **params):
        p = param.ParamOverrides(self, params)

        np.random.seed(p.seed)

        r = 0.5  # radius
        x0, y0 = 0.5, 0.5  # center of unit circle
        circumference = 2 * np.pi

        df = nodes.copy()

        if p.uniform:
            thetas = np.arange(circumference, step=circumference/len(df))
        else:
            thetas = np.asarray(np.random.random((len(df),))) * circumference

        df[p.x] = x0 + r * np.cos(thetas)
        df[p.y] = y0 + r * np.sin(thetas)

        return df


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
            ai = matrix[i].toarray()

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


class forceatlas2_layout(LayoutAlgorithm):
    """
    Assign coordinates to the nodes using force-directed algorithm.

    This is a force-directed graph layout algorithm called
    `ForceAtlas2`.

    Timothee Poisot's `nxfa2` is the original implementation of this
    algorithm.

    .. _ForceAtlas2:
       http://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0098679&type=printable
    .. _nxfa2:
       https://github.com/tpoisot/nxfa2
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

    def _extract_points_from_nodes(self, nodes, dtype=None):
        if self.x in nodes.columns and self.y in nodes.columns:
            points = np.asarray(nodes[[self.x, self.y]])
        else:
            points = np.asarray(np.random.random((len(nodes), self.dim)), dtype=dtype)
        return points

    def _convert_graph_to_sparse_matrix(self, nodes, edges, dtype=None, format='csr'):
        nlen = len(nodes)
        if 'id' in nodes:
            index = dict(zip(nodes['id'].values, range(nlen)))
        else:
            index = dict(zip(nodes.index.values, range(nlen)))

        if self.weight and self.weight in edges:
            edge_values = edges[[self.source, self.target, self.weight]].values
            rows, cols, data = zip(*((index[src], index[dst], weight)
                                     for src, dst, weight in edge_values
                                     if src in index and dst in index))
        else:
            edge_values = edges[[self.source, self.target]].values
            rows, cols, data = zip(*((index[src], index[dst], 1)
                                     for src, dst in edge_values
                                     if src in index and dst in index))

        # Symmetrize matrix
        d = data + data
        r = rows + cols
        c = cols + rows

        # Check for nodes pointing to themselves
        loops = edges[edges[self.source] == edges[self.target]]
        if len(loops):
            if self.weight and self.weight in edges:
                loop_values = loops[[self.source, self.target, self.weight]].values
                diag_index, diag_data = zip(*((index[src], -weight)
                                              for src, dst, weight in loop_values
                                              if src in index and dst in index))
            else:
                loop_values = loops[[self.source, self.target]].values
                diag_index, diag_data = zip(*((index[src], -1)
                                              for src, dst in loop_values
                                              if src in index and dst in index))
            d += diag_data
            r += diag_index
            c += diag_index

        M = scipy.sparse.coo_matrix((d, (r, c)), shape=(nlen, nlen), dtype=dtype)
        return M.asformat(format)

    def _merge_points_with_nodes(self, nodes, points):
        n = nodes.copy()
        n[self.x] = points[:, 0]
        n[self.y] = points[:, 1]
        return n

    def __call__(self, nodes, edges, **params):
        # Merge user-provided parameters with local parameters
        p = param.ParamOverrides(self, params)
        for k in p:
            setattr(self, k, p[k])

        np.random.seed(p.seed)

        # Convert graph into sparse adjacency matrix and array of points
        points = self._extract_points_from_nodes(nodes, dtype='f')
        matrix = self._convert_graph_to_sparse_matrix(nodes, edges, dtype='f')

        if p.k is None:
            p.k = np.sqrt(1.0 / len(points))

        # the initial "temperature" is about .1 of domain area (=1x1)
        # this is the largest step allowed in the dynamics.
        temperature = 0.1

        # simple cooling scheme.
        # linearly step down by dt on each iteration so last iteration is size dt.
        cooling(matrix, points, temperature, p)

        # Return the nodes with updated positions
        return self._merge_points_with_nodes(nodes, points)
