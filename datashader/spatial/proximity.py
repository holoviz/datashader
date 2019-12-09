import xarray
import numpy as np
from numba import njit, prange
from math import sqrt

EUCLIDEAN = 0
GREAT_CIRCLE = 1
MANHATTAN = 2
NAN = np.nan


def _distance_metric_mapping():

    DISTANCE_METRICS = {}
    DISTANCE_METRICS['EUCLIDEAN'] = EUCLIDEAN
    DISTANCE_METRICS['GREAT_CIRCLE'] = GREAT_CIRCLE
    DISTANCE_METRICS['MANHATTAN'] = MANHATTAN

    return DISTANCE_METRICS

# create dictionary to map distance metric presented by string and the
# corresponding metric presented by integer.
DISTANCE_METRICS = _distance_metric_mapping()


@njit(nogil=True)
def euclidean_distance(x1, x2, y1, y2):
    """Calculate Euclidean distance between (x1, y1) and (x2, y2).

    Parameters
    ----------
    x1 : float
         x-coordinate of the first point.
    x2: float
        x-coordinate of the second point.
    y1: float
        y-coordinate of the first point.
    y2: float
        y-coordinate of the second point.

    Returns
    -------
    distance: float
    """

    x = x1 - x2
    y = y1 - y2
    return np.sqrt(x * x + y * y)


@njit(nogil=True)
def manhattan_distance(x1, x2, y1, y2):
    """Calculate Manhattan distance between (x1, y1) and (x2, y2).

    Parameters
    ----------
    x1 : float
         x-coordinate of the first point.
    x2: float
        x-coordinate of the second point.
    y1: float
        y-coordinate of the first point.
    y2: float
        y-coordinate of the second point.

    Returns
    -------
    distance: float
    """

    x = x1 - x2
    y = y1 - y2
    return x * x + y * y


@njit(nogil=True)
def great_circle_distance(x1, x2, y1, y2, radius=6378137):
    """Calculate great-circle distance between (x1, y1) and (x2, y2), assuming each point is a longitude, latitude pair.

    Parameters
    ----------
    x1 : float
         x-coordinate (latitude) of the first point.
    x2: float
        x-coordinate (latitude) of the second point.
    y1: float
        y-coordinate (longitude) of the first point.
    y2: float
        y-coordinate (longitude) of the second point.

    Returns
    -------
    distance: float
    """

    if x1 > 180 or x1 < -180:
        raise ValueError('Invalid x-coordinate of the first point.'
                         'Must be in the range [-180, 180]')

    if x2 > 180 or x2 < -180:
        raise ValueError('Invalid x-coordinate of the second point.'
                         'Must be in the range [-180, 180]')

    if y1 > 90 or y1 < -90:
        raise ValueError('Invalid y-coordinate of the first point.'
                         'Must be in the range [-90, 90]')

    if y2 > 90 or y2 < -90:
        raise ValueError('Invalid y-coordinate of the second point.'
                         'Must be in the range [-90, 90]')

    lon1, lat1, lon2, lat2 = (np.radians(y1), np.radians(x1),
                              np.radians(y2), np.radians(x2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 +\
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # earth radius: 6378137
    return radius * 2 * np.arcsin(np.sqrt(a))


@njit(nogil=True)
def _distance(x1, x2, y1, y2, metric):

    if metric == EUCLIDEAN:
        return euclidean_distance(x1, x2, y1, y2)

    if metric == GREAT_CIRCLE:
        return great_circle_distance(x1, x2, y1, y2)

    if metric == MANHATTAN:
        return manhattan_distance(x1, x2, y1, y2)

    return -1.0


@njit(nogil=True)
def _process_proximity_line(source_line, x_coords, y_coords,
                            pan_near_x, pan_near_y, is_forward,
                            line_id, width, max_distance, line_proximity,
                            values, distance_metric):

    # Process proximity for a line of pixels in an image
    #
    # source_line: 1d ndarray, input data
    # pan_near_x:  1d ndarray
    # pan_near_y:  1d ndarray
    # is_forward: boolean, will we loop forward through pixel?
    # line_id: np.int64, index of the source_line in the image
    # width: np.int64, image width. It is the number of pixels in the
    #                source_line
    # max_distance: np.float64, maximum distance considered.
    # line_proximity: 1d numpy array of type np.float64,
    #                         calculated proximity from source_line
    # values: 1d numpy array of type np.uint8,
    #                 A list of target pixel values to measure the distance from.
    #                 If this option is not provided proximity will be computed
    #                 from non-zero pixel values.
    #                 Currently pixel values are internally processed as integers
    # Return: 1d numpy array of type np.float64.
    #          Corresponding proximity of source_line.

    start = width - 1
    end = -1
    step = -1
    if is_forward:
        start = 0
        end = width
        step = 1

    n_values = len(values)
    for pixel in prange(start, end, step):
        is_target = False
        # Is the current pixel a target pixel?
        if n_values == 0:
            is_target = source_line[pixel] != 0
        else:
            for i in prange(n_values):
                if source_line[pixel] == values[i]:
                    is_target = True

        if is_target:
            line_proximity[pixel] = 0.0
            pan_near_x[pixel] = pixel
            pan_near_y[pixel] = line_id
            continue

        # Are we near(er) to the closest target to the above (below) pixel?
        near_distance_square = max_distance ** 2 * 2.0
        if pan_near_x[pixel] != -1:
            # distance_square
            dist_sqr = _distance(x_coords[pan_near_x[pixel]], x_coords[pixel],
                                 y_coords[pan_near_y[pixel]], y_coords[line_id],
                                 distance_metric)
            if dist_sqr < near_distance_square:
                near_distance_square = dist_sqr
            else:
                pan_near_x[pixel] = -1
                pan_near_y[pixel] = -1

        # Are we near(er) to the closest target to the left (right) pixel?
        last = pixel - step
        if pixel != start and pan_near_x[last] != -1:
            dist_sqr = _distance(x_coords[pan_near_x[last]], x_coords[pixel],
                                 y_coords[pan_near_y[last]], y_coords[line_id],
                                 distance_metric)
            if dist_sqr < near_distance_square:
                near_distance_square = dist_sqr
                pan_near_x[pixel] = pan_near_x[last]
                pan_near_y[pixel] = pan_near_y[last]

        #  Are we near(er) to the closest target to the
        #  topright (bottom left) pixel?
        tr = pixel + step
        if tr != end and pan_near_x[tr] != -1:
            dist_sqr = _distance(x_coords[pan_near_x[tr]], x_coords[pixel],
                                 y_coords[pan_near_y[tr]], y_coords[line_id],
                                 distance_metric)
            if dist_sqr < near_distance_square:
                near_distance_square = dist_sqr
                pan_near_x[pixel] = pan_near_x[tr]
                pan_near_y[pixel] = pan_near_y[tr]

        # Update our proximity value.
        if pan_near_x[pixel] != -1 and not np.isnan(source_line[pixel])\
                and max_distance * max_distance >= near_distance_square\
                and (line_proximity[pixel] < 0 or
                     near_distance_square <
                     line_proximity[pixel] * line_proximity[pixel]):
            line_proximity[pixel] = sqrt(near_distance_square)
    return


@njit(nogil=True)
def _proximity(img, x_coords, y_coords, target_values, distance_metric):

    height, width = img.shape

    max_distance = x_coords[width - 1] + y_coords[height - 1]

    pan_near_x = np.zeros(width, dtype=np.int64)
    pan_near_y = np.zeros(width, dtype=np.int64)

    # output of the function
    img_proximity = np.zeros(shape=(height, width), dtype=np.float64)

    # Loop from top to bottom of the image.
    for i in prange(width):
        pan_near_x[i] = -1
        pan_near_y[i] = -1

    scan_line = np.zeros(width, dtype=img.dtype)

    for line in prange(height):
        # Read for target values.
        for i in prange(width):
            scan_line[i] = img[line][i]

        line_proximity = np.zeros(width, dtype=np.float64)
        for i in prange(width):
            line_proximity[i] = -1.0

        # left to right
        _process_proximity_line(scan_line, x_coords, y_coords,
                                pan_near_x, pan_near_y, True, line,
                                width, max_distance, line_proximity,
                                target_values, distance_metric)

        # right to left
        _process_proximity_line(scan_line, x_coords, y_coords,
                                pan_near_x, pan_near_y, False, line,
                                width, max_distance, line_proximity,
                                target_values, distance_metric)

        for i in prange(width):
            img_proximity[line][i] = line_proximity[i]

    # Loop from bottom to top of the image.
    for i in prange(width):
        pan_near_x[i] = -1
        pan_near_y[i] = -1

    for line in prange(height-1, -1, -1):
        # Read first pass proximity.
        for i in prange(width):
            line_proximity[i] = img_proximity[line][i]

        # Read pixel target_values.
        for i in prange(width):
            scan_line[i] = img[line][i]

        # Right to left
        _process_proximity_line(scan_line, x_coords, y_coords,
                                pan_near_x, pan_near_y, False, line,
                                width, max_distance, line_proximity,
                                target_values, distance_metric)

        # Left to right
        _process_proximity_line(scan_line, x_coords, y_coords,
                                pan_near_x, pan_near_y, True, line,
                                width, max_distance, line_proximity,
                                target_values, distance_metric)

        # final post processing of distances
        for i in prange(width):
            if line_proximity[i] < 0 or np.isnan(scan_line[i]):
                # this corresponds the the nan value of input raster.
                line_proximity[i] = np.nan

        for i in prange(width):
            img_proximity[line][i] = line_proximity[i]
    return img_proximity


# ported from
# https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp
def proximity(raster, target_values=[], distance_metric='EUCLIDEAN'):
    """Compute the proximity of all pixels in the image to a set of pixels in
    the source image.

    This function attempts to compute the proximity of all pixels in the
    image to a set of pixels in the source image. The following options are
    used to define the behavior of the function. By default all non-zero pixels
    in ``raster.values`` will be considered the "target", and all proximities
    will be computed in pixels.  Note that target pixels are set to the value
    corresponding to a distance of zero.

    Parameters
    ----------
    raster: xarray.DataArray
        Input raster image with shape=(height, width)
    target_values: list
        Target pixel values to measure the distance from.  If this option is
        not provided, proximity will be computed from non-zero pixel values.
        Currently pixel values are internally processed as integers.
    distance_metric: string
        The metric for calculating distance between 2 points.
        Valid distance_metrics include: 'EUCLIDEAN', 'GREAT_CIRCLE', and 'MANHATTAN'
        Default is 'EUCLIDEAN'.

    Returns
    -------
    proximity: xarray.DataArray
        Proximity image with shape=(height, width)
    """

    raster_dims = raster.dims
    if raster_dims != ('y', 'x'):
        raise ValueError("In function proximity(). "
                         "raster.coords should be named as coordinates:"
                         "('y', 'x')")

    # convert distance metric from string to integer, the correct type
    # of argument for function _distance()
    distance_metric = DISTANCE_METRICS.get(distance_metric, None)

    if distance_metric is None:
        distance_metric = DISTANCE_METRICS['EUCLIDEAN']

    target_values = np.asarray(target_values).astype(np.uint8)

    img = raster.values
    y_coords = raster.coords['y'].values
    x_coords = raster.coords['x'].values

    proximity_img =  _proximity(img, x_coords, y_coords, target_values,
                                distance_metric)

    result = xarray.DataArray(proximity_img,
                              coords=raster.coords,
                              dims=raster.dims,
                              attrs=raster.attrs)

    return result
