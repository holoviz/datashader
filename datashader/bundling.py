from __future__ import absolute_import, division, print_function

from math import ceil

from dask import compute, delayed
from pandas import DataFrame
from skimage.filters import gaussian, sobel_h, sobel_v

import numba as nb
import numpy as np
import pandas as pd

from .utils import ngjit


ACCURACY = 500

MIN_SEG = 0.008
MAX_SEG = 0.016
TENSION = 0.3


@ngjit
def distance_between(a, b):
    return (((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))**(0.5)


@nb.jit
def resample_segment(segments, new_segments, min_segment_length=MIN_SEG, max_segment_length=MAX_SEG):
    next_point = np.array([0.0, 0.0])
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
def calculate_length(segments, min_segment_length=MIN_SEG, max_segment_length=MAX_SEG):
    next_point = np.array([0.0, 0.0])
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


def resample_edge(segments, min_segment_length=MIN_SEG, max_segment_length=MAX_SEG):
    change, total_resamples = calculate_length(segments, min_segment_length, max_segment_length)
    if not change:
        return segments
    resampled = np.empty((total_resamples, 2))
    resample_segment(segments, resampled, min_segment_length, max_segment_length)
    return resampled


@delayed
def resample_edges(edge_segments, min_segment_length=MIN_SEG, max_segment_length=MAX_SEG):
    replaced_edges = []
    for segments in edge_segments:
        replaced_edges.append(resample_edge(segments, min_segment_length, max_segment_length))
    return replaced_edges


@nb.jit
def smooth_segment(segments, tension):
    seg_length = len(segments) - 2
    for i in range(1, seg_length):
        previous, current, next_point = segments[i - 1], segments[i], segments[i + 1]
        current[0] = ((1 - tension) * current[0]) + (tension * (previous[0] + next_point[0]) / 2)
        current[1] = ((1 - tension) * current[1]) + (tension * (previous[1] + next_point[1]) / 2)


@nb.jit
def smooth(edge_segments, tension):
    for segments in edge_segments:
        smooth_segment(segments, tension)


@ngjit
def advect_segments(segments, vert, horiz):
    for i in range(1, len(segments) - 1):
        x = int(segments[i][0] * ACCURACY)
        y = int(segments[i][1] * ACCURACY)
        segments[i][0] = segments[i][0] + horiz[x, y] / ACCURACY
        segments[i][1] = segments[i][1] + vert[x, y] / ACCURACY
        segments[i][0] = max(0, min(segments[i][0], 1))
        segments[i][1] = max(0, min(segments[i][1], 1))


def advect_and_resample(vert, horiz, segments, iterations=50):
    for it in range(iterations):
        advect_segments(segments, vert, horiz)
        if it % 2 == 0:
            segments = resample_edge(segments)
    return segments


@delayed
def advect_resample_all(gradients, edge_segments):
    vert, horiz = gradients
    return [advect_and_resample(vert, horiz, edges) for edges in edge_segments]


def batches(l, n):
    """Yield successive n-sized batches from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


@delayed
def draw_to_surface(edge_segments, bandwidth):
    img = np.zeros((ACCURACY + 1, ACCURACY + 1))
    for segments in edge_segments:
        for point in segments:
            img[int(point[0] * ACCURACY), int(point[1] * ACCURACY)] += 1
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


def _convert_graph_to_edge_segments(nodes, edges):
    def minmax_scale(series):
        minimum, maximum = np.min(series), np.max(series)
        return (series - minimum) / (maximum - minimum)

    nodes['x'] = minmax_scale(nodes['x'])
    nodes['y'] = minmax_scale(nodes['y'])

    df = pd.merge(nodes, edges, left_index=True, right_on=['source'])
    df = df.rename(columns={'x': 'src_x', 'y': 'src_y'})

    df = pd.merge(nodes, df, left_index=True, right_on=['target'])
    df = df.rename(columns={'x': 'dst_x', 'y': 'dst_y'})

    df = df.filter(items=['src_x', 'src_y', 'dst_x', 'dst_y'])

    edge_segments = []
    for edge in df.get_values():
        segments = [[edge[0], edge[1]], [edge[2], edge[3]]]
        edge_segments.append(np.array(segments))
    return edge_segments


def _convert_edge_segments_to_dataframe(edge_segments):
    # Need to put a [np.nan, np.nan] between edges
    def edge_iterator():
        for edge in edge_segments:
            yield edge
            yield np.array([[np.nan, np.nan]])

    df = DataFrame(np.concatenate(list(edge_iterator())))
    df.columns = ['x', 'y']
    return df



import param

class directly_connect_edges(param.ParameterizedFunction):
    """
    Convert a graph into paths suitable for datashading.

    Base class that connects each edge using a single line segment.
    Subclasses can add more complex algorithms for connecting with
    curved or manhattan-style polylines.
    """

    def __call__(self, nodes, edges):
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
        edges = _convert_graph_to_edge_segments(nodes, edges)
        return _convert_edge_segments_to_dataframe(edges)


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

    def __call__(self, nodes, edges, **params):
        p = param.ParamOverrides(self,params)

        # Convert graph into list of edge segments
        edges = _convert_graph_to_edge_segments(nodes, edges)
    
        # This is simply to let the work split out over multiple cores
        edge_batches = list(batches(edges, p.batch_size))
    
        # This gets the edges split into lots of small segments
        # Doing this inside a delayed function lowers the transmission overhead
        edge_segments = [resample_edges(batch) for batch in edge_batches]
    
        for i in range(p.iterations):
            # Each step, the size of the 'blur' shrinks
            bandwidth = p.initial_bandwidth * p.decay**(i + 1) * ACCURACY
    
            # If it's this small, there won't be a change anyway
            if bandwidth < 2:
                break
    
            # Draw the density maps and combine them
            images = [draw_to_surface(segment, bandwidth) for segment in edge_segments]
            overall_image = sum(images)
    
            gradients = get_gradients(overall_image)
    
            # Move edges along the gradients and resample when necessary
            # This could include smoothing to adjust the amount a graph can change
            edge_segments = [advect_resample_all(gradients, segment) for segment in edge_segments]
    
        # Do a final resample to a smaller size for nicer rendering
        edge_segments = [resample_edges(segment, MIN_SEG / 2, MAX_SEG / 2) for segment in edge_segments]
    
        # Finally things can be sent for computation
        edge_segments = compute(edge_segments)[0]
    
        # Smooth out the graph
        for i in range(10):
            for batch in edge_segments:
                smooth(batch, TENSION)
    
        # Flatten things
        new_segs = []
        for batch in edge_segments:
            new_segs.extend(batch)
    
        # Convert list of edge segments to Pandas dataframe
        return _convert_edge_segments_to_dataframe(new_segs)
