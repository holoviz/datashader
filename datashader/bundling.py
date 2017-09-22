"""Bundle a graph's edges to emphasize the graph structure.

Given a large graph, the underlying structure can be obscured by edges in close
proximity. To uncover the group structure for clearer visualization, edges are
split into smaller edges and bundled with neighbors.

Ian Calvert's `Edgehammer`_ is the original implementation of the main
algorithm.

.. _Edgehammer:
   https://gitlab.com/ianjcalvert/edgehammer
"""

from __future__ import absolute_import, division, print_function

from math import ceil

from dask import compute, delayed
from pandas import DataFrame
from skimage.filters import gaussian, sobel_h, sobel_v

import numba as nb
import numpy as np
import pandas as pd
import param

from .utils import ngjit


@ngjit
def distance_between(a, b):
    """Find the Euclidean distance between two points."""
    return (((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))**(0.5)


@nb.jit
def resample_segment(segments, new_segments, min_segment_length, max_segment_length, segment_class):
    next_point = segment_class.create_point()
    current_point = segments[0]
    pos = 0
    index = 1
    while index < len(segments):
        next_point = segments[index]
        distance = distance_between(current_point, next_point)
        if (distance < min_segment_length and 1 < index < (len(segments) - 1)):
            # Merge points, because they're too close to each other
            current_point = (current_point + next_point) / 2
            new_segments[pos] = current_point
            pos += 1
            index += 2
        elif distance > max_segment_length:
            # If points are too far away from each other, linearly place new points
            points = ceil(distance / ((max_segment_length + min_segment_length) / 2))
            for i in range(points):
                new_segments[pos] = current_point + (i * ((next_point - current_point) / points))
                pos += 1
            current_point = next_point
            index += 1
        else:
            # Do nothing, everything is good
            new_segments[pos] = current_point
            pos += 1
            current_point = next_point
            index += 1
    new_segments[pos] = next_point
    return new_segments


@nb.jit
def calculate_length(segments, min_segment_length, max_segment_length, segment_class):
    current_point = segments[0]
    index = 1
    total = 0
    any_change = False
    while index < len(segments):
        next_point = segments[index]
        distance = distance_between(current_point, next_point)
        if (distance < min_segment_length and 1 < index < (len(segments) - 1)):
            any_change = True
            current_point = (current_point + next_point) / 2
            total += 1
            index += 2
        elif distance > max_segment_length:
            any_change = True
            # Linear subsample
            points = ceil(distance / ((max_segment_length + min_segment_length) / 2))
            total += points
            current_point = next_point
            index += 1
        else:
            # Do nothing
            total += 1
            current_point = next_point
            index += 1
    total += 1
    return any_change, total


def resample_edge(segments, min_segment_length, max_segment_length, segment_class):
    change, total_resamples = calculate_length(segments, min_segment_length, max_segment_length, segment_class)
    if not change:
        return segments
    resampled = segment_class.create_empty_points(total_resamples)
    resample_segment(segments, resampled, min_segment_length, max_segment_length, segment_class)
    return resampled


@delayed
def resample_edges(edge_segments, min_segment_length, max_segment_length, segment_class):
    replaced_edges = []
    for segments in edge_segments:
        replaced_edges.append(resample_edge(segments, min_segment_length, max_segment_length, segment_class))
    return replaced_edges


@nb.jit
def smooth_segment(segments, tension):
    seg_length = len(segments) - 2
    for i in range(1, seg_length):
        previous, current, next_point = segments[i - 1], segments[i], segments[i + 1]
        current[1] = ((1 - tension) * current[1]) + (tension * (previous[1] + next_point[1]) / 2)
        current[2] = ((1 - tension) * current[2]) + (tension * (previous[2] + next_point[2]) / 2)


@nb.jit
def smooth(edge_segments, tension):
    for segments in edge_segments:
        smooth_segment(segments, tension)


@ngjit
def advect_segments(segments, vert, horiz, accuracy):
    for i in range(1, len(segments) - 1):
        x = int(segments[i][1] * accuracy)
        y = int(segments[i][2] * accuracy)
        segments[i][1] = segments[i][1] + horiz[x, y] / accuracy
        segments[i][2] = segments[i][2] + vert[x, y] / accuracy
        segments[i][1] = max(0, min(segments[i][1], 1))
        segments[i][2] = max(0, min(segments[i][2], 1))


def advect_and_resample(vert, horiz, segments, iterations, accuracy, min_segment_length, max_segment_length, segment_class):
    for it in range(iterations):
        advect_segments(segments, vert, horiz, accuracy)
        if it % 2 == 0:
            segments = resample_edge(segments, min_segment_length, max_segment_length, segment_class)
    return segments


@delayed
def advect_resample_all(gradients, edge_segments, iterations, accuracy, min_segment_length, max_segment_length, segment_class):
    vert, horiz = gradients
    return [advect_and_resample(vert, horiz, edges, iterations, accuracy, min_segment_length, max_segment_length, segment_class)
            for edges in edge_segments]


def batches(l, n):
    """Yield successive n-sized batches from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


@delayed
def draw_to_surface(edge_segments, bandwidth, accuracy, accumulator):
    img = np.zeros((accuracy + 1, accuracy + 1))
    for segments in edge_segments:
        for point in segments:
            accumulator(img, point, accuracy)
    return gaussian(img, sigma=bandwidth / 2)


@delayed
def get_gradients(img):
    img /= np.max(img)

    horiz = sobel_h(img)
    vert = sobel_v(img)

    magnitude = np.sqrt(horiz**2 + vert**2) + 1e-5
    vert /= magnitude
    horiz /= magnitude
    return (vert, horiz)


class BaseSegment(object):
    @classmethod
    @nb.jit
    def create_point(cls):
        return np.array([0.0] * cls.ndims)

    @classmethod
    @nb.jit
    def create_empty_points(cls, n):
        return np.empty((n, cls.ndims))

    @classmethod
    @nb.jit
    def create_delimiter(cls):
        return np.array([[np.nan] * cls.ndims])


class UnweightedSegment(BaseSegment):
    ndims = 3
    columns = ['edge_id', 'x', 'y']
    merged_columns = ['edge_id', 'src_x', 'src_y', 'dst_x', 'dst_y']

    @staticmethod
    @nb.jit
    def create_segment(edge):
        return np.array([[edge[0], edge[1], edge[2]], [edge[0], edge[3], edge[4]]])

    @staticmethod
    @ngjit
    def accumulate(img, point, accuracy):
        img[int(point[1] * accuracy), int(point[2] * accuracy)] += 1


class EdgelessUnweightedSegment(BaseSegment):
    ndims = 2
    columns = ['x', 'y']
    merged_columns = ['src_x', 'src_y', 'dst_x', 'dst_y']

    @staticmethod
    @nb.jit
    def create_segment(edge):
        return np.array([[edge[0], edge[1]], [edge[2], edge[3]]])

    @staticmethod
    @ngjit
    def accumulate(img, point, accuracy):
        img[int(point[0] * accuracy), int(point[1] * accuracy)] += 1


class WeightedSegment(BaseSegment):
    ndims = 4
    columns = ['edge_id', 'x', 'y', 'weight']
    merged_columns = ['edge_id', 'src_x', 'src_y', 'dst_x', 'dst_y', 'weight']

    @staticmethod
    @nb.jit
    def create_segment(edge):
        return np.array([[edge[0], edge[1], edge[2], edge[5]], [edge[0], edge[3], edge[4], edge[5]]])

    @staticmethod
    @ngjit
    def accumulate(img, point, accuracy):
        img[int(point[1] * accuracy), int(point[2] * accuracy)] += point[3]


class EdgelessWeightedSegment(BaseSegment):
    ndims = 3
    columns = ['x', 'y', 'weight']
    merged_columns = ['src_x', 'src_y', 'dst_x', 'dst_y', 'weight']

    @staticmethod
    @nb.jit
    def create_segment(edge):
        return np.array([[edge[0], edge[1], edge[4]], [edge[2], edge[3], edge[4]]])

    @staticmethod
    @ngjit
    def accumulate(img, point, accuracy):
        img[int(point[0] * accuracy), int(point[1] * accuracy)] += point[2]


def _convert_graph_to_edge_segments(nodes, edges, include_edge_id, ignore_weights=False):
    """
    Merge graph dataframes into a list of edge segments.

    Given a graph defined as a pair of dataframes (nodes and edges), the
    nodes (id, coordinates) and edges (id, source, target, weight) are
    joined by node id to create a single dataframe with each source/target
    of an edge (including its optional weight) replaced with the respective
    coordinates. For both nodes and edges, each id column is assumed to be
    the index.

    We also return the dimensions of each point in the final dataframe and
    the accumulator function for drawing to an image.
    """

    df = pd.merge(edges, nodes, left_on=['source'], right_index=True)
    df = df.rename(columns={'x': 'src_x', 'y': 'src_y'})

    df = pd.merge(df, nodes, left_on=['target'], right_index=True)
    df = df.rename(columns={'x': 'dst_x', 'y': 'dst_y'})

    df = df.sort_index()
    df = df.reset_index()

    if include_edge_id:
        df = df.rename(columns={'id': 'edge_id'})

    include_weight = not ignore_weights and 'weight' in edges

    if include_edge_id:
        if include_weight:
            segment_class = WeightedSegment
        else:
            segment_class = UnweightedSegment
    else:
        if include_weight:
            segment_class = EdgelessWeightedSegment
        else:
            segment_class = EdgelessUnweightedSegment

    df = df.filter(items=segment_class.merged_columns)

    edge_segments = []
    for edge in df.get_values():
        edge_segments.append(segment_class.create_segment(edge))
    return edge_segments, segment_class


def _convert_edge_segments_to_dataframe(edge_segments, segment_class):
    """
    Convert list of edge segments into a dataframe.

    For all edge segments, we create a dataframe to represent a path
    as successive points separated by a point with NaN as the x or y
    value.
    """

    # Need to put an array of NaNs with size point_dims between edges
    def edge_iterator():
        for edge in edge_segments:
            yield edge
            yield segment_class.create_delimiter()

    df = DataFrame(np.concatenate(list(edge_iterator())))
    df.columns = segment_class.columns
    return df


class directly_connect_edges(param.ParameterizedFunction):
    """
    Convert a graph into paths suitable for datashading.

    Base class that connects each edge using a single line segment.
    Subclasses can add more complex algorithms for connecting with
    curved or manhattan-style polylines.
    """

    include_edge_id = param.Boolean(default=False, doc="""
        Include edge IDs in bundled dataframe""")

    def __call__(self, nodes, edges, **params):
        """
        Convert a graph data structure into a path structure for plotting

        Given a set of nodes (as a dataframe with a unique ID for each
        node) and a set of edges (as a dataframe with with columns for the
        source and destination IDs for each edge), returns a dataframe
        with with one path for each edge suitable for use with
        Datashader. The returned dataframe has columns for x and y
        location, with paths represented as successive points separated by
        a point with NaN as the x or y value.
        """
        p = param.ParamOverrides(self, params)
        edges, segment_class = _convert_graph_to_edge_segments(nodes, edges, p.include_edge_id, ignore_weights=True)
        return _convert_edge_segments_to_dataframe(edges, segment_class)


@nb.jit
def minmax_normalize(X, lower, upper):
    return (X - lower) / (upper - lower)


@nb.jit
def minmax_denormalize(X, lower, upper):
    return X * (upper - lower) + lower


class hammer_bundle(directly_connect_edges):
    """
    Iteratively group edges and return as paths suitable for datashading.

    Breaks each edge into a path with multiple line segments, and
    iteratively curves this path to bundle edges into groups.
    """

    initial_bandwidth = param.Number(default=0.05,bounds=(0.0,None),doc="""
        Initial value of the bandwidth....""")

    decay = param.Number(default=0.7,bounds=(0.0,1.0),doc="""
        Rate of decay in the bandwidth value, with 1.0 indicating no decay.""")

    iterations = param.Integer(default=4,bounds=(1,None),doc="""
        Number of passes for the smoothing algorithm""")

    batch_size = param.Integer(default=20000,bounds=(1,None),doc="""
        Number of edges to process together""")

    tension = param.Number(default=0.3,bounds=(0,None),precedence=-0.5,doc="""
        Exponential smoothing factor to use when smoothing""")

    accuracy = param.Integer(default=500,bounds=(1,None),precedence=-0.5,doc="""
        Number of entries in table for...""")

    advect_iterations = param.Integer(default=50,bounds=(0,None),precedence=-0.5,doc="""
        Number of iterations to move edges along gradients""")

    min_segment_length = param.Number(default=0.008,bounds=(0,None),precedence=-0.5,doc="""
        Minimum length (in data space?) for an edge segment""")

    max_segment_length = param.Number(default=0.016,bounds=(0,None),precedence=-0.5,doc="""
        Maximum length (in data space?) for an edge segment""")

    include_edge_id = param.Boolean(default=False, doc="""
        Include edge IDs in bundled dataframe""")

    def __call__(self, nodes, edges, **params):
        p = param.ParamOverrides(self, params)

        # Calculate min/max for coordinates
        xmin, xmax = np.min(nodes['x']), np.max(nodes['x'])
        ymin, ymax = np.min(nodes['y']), np.max(nodes['y'])

        # Normalize coordinates
        nodes = nodes.copy()
        nodes['x'] = minmax_normalize(nodes['x'], xmin, xmax)
        nodes['y'] = minmax_normalize(nodes['y'], ymin, ymax)

        # Convert graph into list of edge segments
        edges, segment_class = _convert_graph_to_edge_segments(nodes, edges, p.include_edge_id)

        # This is simply to let the work split out over multiple cores
        edge_batches = list(batches(edges, p.batch_size))

        # This gets the edges split into lots of small segments
        # Doing this inside a delayed function lowers the transmission overhead
        edge_segments = [resample_edges(batch, p.min_segment_length, p.max_segment_length, segment_class) for batch in edge_batches]

        for i in range(p.iterations):
            # Each step, the size of the 'blur' shrinks
            bandwidth = p.initial_bandwidth * p.decay**(i + 1) * p.accuracy

            # If it's this small, there won't be a change anyway
            if bandwidth < 2:
                break

            # Draw the density maps and combine them
            images = [draw_to_surface(segment, bandwidth, p.accuracy, segment_class.accumulate) for segment in edge_segments]
            overall_image = sum(images)

            gradients = get_gradients(overall_image)

            # Move edges along the gradients and resample when necessary
            # This could include smoothing to adjust the amount a graph can change
            edge_segments = [advect_resample_all(gradients, segment, p.advect_iterations, p.accuracy, p.min_segment_length, p.max_segment_length, segment_class)
                             for segment in edge_segments]

        # Do a final resample to a smaller size for nicer rendering
        edge_segments = [resample_edges(segment, p.min_segment_length, p.max_segment_length, segment_class) for segment in edge_segments]

        # Finally things can be sent for computation
        edge_segments = compute(*edge_segments)

        # Smooth out the graph
        for i in range(10):
            for batch in edge_segments:
                smooth(batch, p.tension)

        # Flatten things
        new_segs = []
        for batch in edge_segments:
            new_segs.extend(batch)

        # Convert list of edge segments to Pandas dataframe
        df = _convert_edge_segments_to_dataframe(new_segs, segment_class)

        # Denormalize coordinates
        df['x'] = minmax_denormalize(df['x'], xmin, xmax)
        df['y'] = minmax_denormalize(df['y'], ymin, ymax)

        return df
