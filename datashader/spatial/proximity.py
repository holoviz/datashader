import xarray
import numpy as np
import numba as nb
from numba import jit, prange
import math
from math import sqrt
import warnings

SQUARE_DISTANCE = 0
DISTANCE_METRICS = [SQUARE_DISTANCE]
DISTANCE_METRICS_STR = ['square distance']

@jit(nb.f8(nb.f8, nb.f8, nb.f8, nb.f8, nb.i8), nopython=True)
def distance(x1, x2, y1, y2, metric):
    """
    Calculate distance between (x1, y1) and (x2, y2) using a
        specific distance metric.
    :param x1: np.float64, x-coordinate of the first point.
    :param x2: np.float64, x-coordinate of the second point.
    :param y1: np.float64, y-coordinate of the first point.
    :param y2: np.float64, y-coordinate of the second point.
    :param metric: np.int64, metric to be used for calculating distance.
    :return: np.float64, distance between (x1, y1) and (x2, y2)
             If the metric is not recognized, return -1.0
    """
    if metric == SQUARE_DISTANCE:
        x = x1 - x2
        y = y1 - y2
        return x * x + y * y
    return -1.0


@jit(nb.void(nb.u1[:], nb.i8[:], nb.i8[:], nb.b1, nb.i8, nb.i8, nb.f8, nb.f8,
             nb.f8[:], nb.u1[:], nb.i8), nopython=True)
def process_proximity_line(source_line, pan_near_x, pan_near_y, is_forward,
                           line_id, width, min_distance, max_distance,
                           line_proximity, values, distance_metric):
    """
    Process proximity for a line of pixels in an image

    :param source_line: 1d ndarray of type np.uint8, input data
    :param pan_near_x:  1d ndarray of type np.int64
    :param pan_near_y:  1d ndarray of type np.int64
    :param is_forward: boolean, will we loop forward through pixel?
    :param line_id: np.int64, index of the source_line in the image
    :param width: np.int64, image width. It is the number of pixels in the
                    source_line
    :param max_distance: np.float64, maximum distance considered.
    :param line_proximity: 1d numpy array of type np.float64,
                            calculated proximity from source_line
    :param values: 1d numpy array of type np.uint8,
                    A list of target pixel values to measure the distance from.
                    If this option is not provided proximity will be computed
                    from non-zero pixel values.
                    Currently pixel values are internally processed as integers
    :return: 1d numpy array of type np.float64.
             Corresponding proximity of source_line.
    """

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
        near_distance_square = max(max_distance, width) ** 2 * 2.0
        if pan_near_x[pixel] != -1:
            distance_square = distance(pan_near_x[pixel], pixel,
                                       pan_near_y[pixel], line_id,
                                       distance_metric)
            if distance_square < near_distance_square:
                near_distance_square = distance_square
            else:
                pan_near_x[pixel] = -1
                pan_near_y[pixel] = -1

        # Are we near(er) to the closest target to the left (right) pixel?
        last = pixel - step
        if pixel != start and pan_near_x[last] != -1:
            distance_square = distance(pan_near_x[last], pixel,
                                       pan_near_y[last], line_id,
                                       distance_metric)
            if distance_square < near_distance_square:
                near_distance_square = distance_square
                pan_near_x[pixel] = pan_near_x[last]
                pan_near_y[pixel] = pan_near_y[last]

        #  Are we near(er) to the closest target to the
        #  topright (bottom left) pixel?
        tr = pixel + step
        if tr != end and pan_near_x[tr] != -1:
            distance_square = distance(pan_near_x[tr], pixel,
                                       pan_near_y[tr], line_id,
                                       distance_metric)
            if distance_square < near_distance_square:
                near_distance_square = distance_square
                pan_near_x[pixel] = pan_near_x[tr]
                pan_near_y[pixel] = pan_near_y[tr]

        # Update our proximity value.
        if pan_near_x[pixel] != -1 \
                and max_distance * max_distance >= near_distance_square\
                >= min_distance * min_distance \
                and (line_proximity[pixel] < 0 or
                     near_distance_square <
                     line_proximity[pixel] * line_proximity[pixel]):
            line_proximity[pixel] = sqrt(near_distance_square)
    return


@jit(nb.f8[:, :](nb.u1[:, :], nb.f8, nb.f8, nb.u1[:], nb.f8, nb.i8),
     nopython=True)
def _proximity(img, min_distance, max_distance, target_values, nodata,
               distance_metric):
    """
    Implementation of proximity()
    :param img: 2D numpy array input image of type np.unit8
    :param min_distance: np.float64, minimum distance to search.
                            Proximity distances less than this value will not
                            be computed. Instead output pixels will be set to
                            nodata value.
    :param max_distance: np.float64, maximum distance to search.
                            Proximity distances greater than this value
                            will not be computed. Instead output pixels will be
                            set to nodata value.
    :param nodata: The NODATA value to use on the output band for pixels that
                        are beyond [min_distance, max_distance].
    :param distance_metric: The metric for calculating distance between
                            2 points. Default is square distance.

    :return: 2D numpy array of type np.float64 that represents the
                        proximity image with shape=(height, width)
    """
    height, width = img.shape
    pan_near_x = np.zeros(width, dtype=np.int64)
    pan_near_y = np.zeros(width, dtype=np.int64)

    # output of the function
    img_proximity = np.zeros(shape=(height, width), dtype=np.float64)

    # Loop from top to bottom of the image.
    for i in prange(width):
        pan_near_x[i] = -1
        pan_near_y[i] = -1

    scan_line = np.zeros(width, dtype=np.uint8)
    for line in prange(height):
        # Read for target values.
        for i in prange(width):
            scan_line[i] = img[line][i]

        line_proximity = np.zeros(width, dtype=np.float64)
        for i in prange(width):
            line_proximity[i] = -1.0

        # left to right
        process_proximity_line(scan_line, pan_near_x, pan_near_y, True, line,
                               width, min_distance, max_distance,
                               line_proximity, target_values, distance_metric)

        # right to left
        process_proximity_line(scan_line, pan_near_x, pan_near_y, False, line,
                               width, min_distance, max_distance,
                               line_proximity, target_values, distance_metric)

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
        process_proximity_line(scan_line, pan_near_x, pan_near_y, False, line,
                               width, min_distance, max_distance,
                               line_proximity, target_values, distance_metric)

        # Left to right
        process_proximity_line(scan_line, pan_near_x, pan_near_y, True, line,
                               width, min_distance, max_distance,
                               line_proximity, target_values, distance_metric)

        # final post processing of distances
        for i in prange(width):
            if line_proximity[i] < 0.0:
                # beyond max_distance
                line_proximity[i] = nodata
            elif line_proximity[i] > 0:
                line_proximity[i] = line_proximity[i] * 1.0

        for i in prange(width):
            img_proximity[line][i] = line_proximity[i]
    return img_proximity


# ported from
# https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp
def proximity(raster, min_distance=None, max_distance=None, target_values=[],
              nodata=np.nan, distance_metric=None):
    """
        Compute the proximity of all pixels in the image to a set of pixels in
        the source image.

        This function attempts to compute the proximity of all pixels in
        the image to a set of pixels in the source image.  The following
        options are used to define the behavior of the function.  By
        default all non-zero pixels in hSrcBand will be considered the
        "target", and all proximities will be computed in pixels.  Note
        that target pixels are set to the value corresponding to a distance
        of zero.
        :param img: 2D ndarray that represents input raster image with
                        shape=(height, width)
        :param target_values: A list of target pixel values to measure
                        the distance from.  If this option is not provided,
                        proximity will be computed from non-zero pixel values.
                        Currently pixel values are internally processed
                        as integers.
        :param min_distance: The minumum distance to search.
                            Proximity distances less than this value will not
                            be computed.  Instead output pixels will be set to
                            a nodata value.
        :param max_distance: The maximum distance to search.
                            Proximity distances greater than this value will
                            not be computed.  Instead output pixels will be
                            set to a nodata value.
        :param nodata: The NODATA value to use on the output band for pixels
                        that are beyond [min_distance, max_distance].
        :param distance_metric: The metric for calculating distance
                                between 2 points. Default is square distance.

        :return: 2D ndarray that represents the proximity image with
                shape=(height, width)

        Examples
        --------

        img = = np.array([[3, 3, 0, 1, 2],
                          [2, 0, 0, 0, 4],
                          [2, 0, 0, 0, 1],
                          [2, 0, 0, 0, 1],
                          [2, 0, 0, 0, 1]], dtype=np.uint8)

        # caculate proximity using default setting
        prox = proximity(img)
        # Result:
        array([[0.        , 0.        , 1.        , 0.        , 0.        ],
               [0.        , 1.        , 1.41421356, 1.        , 0.        ],
               [0.        , 1.        , 2.        , 1.        , 0.        ],
               [0.        , 1.        , 2.        , 1.        , 0.        ],
               [0.        , 1.        , 2.        , 1.        , 0.        ]])

        # calculate proximity for specific target values
        prox = proximity(img, target_values=[3, 4])
        # Result:
        array([[0.        , 0.        , 1.        , 1.41421356, 1.        ],
               [1.        , 1.        , 1.41421356, 1.        , 0.        ],
               [2.        , 2.        , 2.23606798, 1.41421356, 1.        ],
               [3.        , 3.        , 2.82842712, 2.23606798, 2.        ],
               [4.        , 4.        , 3.60555128, 3.16227766, 3.        ]])

        # calculate proximity within a max distance
        prox = proximity(img, target_values=[3, 4], max_distance=3)
        # Result:
        array([[0.        , 0.        , 1.        , 1.41421356, 1.        ],
               [1.        , 1.        , 1.41421356, 1.        , 0.        ],
               [2.        , 2.        , 2.23606798, 1.41421356, 1.        ],
               [3.        , 3.        , 2.82842712, 2.23606798, 2.        ],
               [       nan,        nan,        nan,        nan, 3.        ]])

        # calculate proximity within a max distance, set pixels that beyond
        # the max distance to a specific value
        prox = proximity(img, target_values=[3, 4], max_distance=3, nodata=-1)
        # Result
        array([[ 0.        ,  0.        ,  1.        ,  1.41421356,  1.     ],
               [ 1.        ,  1.        ,  1.41421356,  1.        ,  0.     ],
               [ 2.        ,  2.        ,  2.23606798,  1.41421356,  1.     ],
               [ 3.        ,  3.        ,  2.82842712,  2.23606798,  2.     ],
               [-1.        , -1.        , -1.        , -1.        ,  3.     ]])
        """

    if not distance_metric:
        warnings.warn("No distance metric specified. "
                      "Using square distance for calculating proximity.")
        distance_metric = SQUARE_DISTANCE

    if not callable(DISTANCE_METRICS[distance_metric]):
        warnings.warn("Invalid distance metric. "
                      "Using square distance for calculating proximity.")
        distance_metric = SQUARE_DISTANCE

    img = raster.values
    height, width = img.shape
    if max_distance is None:
        max_distance = height + width

    if min_distance is None:
        min_distance = 0

    min_distance *= 1.0
    max_distance *= 1.0

    if max_distance < min_distance:
        raise ValueError("min_distance must not exceed max_distance.")

    if not len(target_values):
        warnings.warn("No target value specified. "
                      "Calculate proximity from non-zero pixels.")
        distance_metric = SQUARE_DISTANCE

    target_values = np.asarray(target_values).astype(np.uint8)
    nodata = np.float64(nodata)

    proximity_img =  _proximity(img, min_distance, max_distance, target_values,
                                nodata, distance_metric)

    result = xarray.DataArray(proximity_img,
                              coords=raster.coords,
                              dims=raster.dims)
    
    result.attrs['min_distance'] = min_distance
    result.attrs['max_distance'] = max_distance 
    result.attrs['target_values'] = target_values 
    result.attrs['nodata_value'] = nodata
    result.attrs['distance_metric'] = DISTANCE_METRICS_STR[distance_metric]

    return result
