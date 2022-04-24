"""Tests for antialiased line drawing in Datashader"""

# The single-pixel width tests consist of 6 test cases, each drawing a
# different set of lines on a square canvas. Tests 1 to 4 are all sets of lines
# that start from the same point of origin and then disperse across the canvas
# to fill either an upper or a lower triangle. What follows is a simple
# ascii-art based cartoon to illustrate the setup. It only has 3 lines instead
# of nine and it is a cartoon within the limits of being able to draw angled
# lines on a square grid. The origin is the lower left corner and all nine
# lines are in the triangle below the diagonal.
#
# +---------------------------------------------+
# |                                             |
# |                                         *   |
# |                                       *     |
# |                                     *       |
# |                                   *         |
# |                                 *           |
# |                               *             |
# |                             *               |
# |                           *                 |
# |                         *                   |
# |                       *                     |
# |                     *                    *  |
# |                   *                  *      |
# |                 *                *          |
# |               *              *              |
# |             *            *                  |
# |           *          *                      |
# |         *        *                          |
# |       *      *                              |
# |     *    *                                  |
# |   *  *                                      |
# | ******************************************* |
# |                                             |
# +---------------------------------------------+
#
# This is test_001. The following three tests 002, 003 and 004 are:
# * test_002: lower left corner as origin, lines in upper triangle
# * test_003: upper right corner as origin, lines in upper triangle
# * test_004: upper right corner as origin, lines in lower triangle
#
# The tests 005 and 006 use a different pattern, which is a combination of
# vertical and diagonal lines, where the angle of the diagonal lines
# 'increases'. Here is another cartoon to illustrate these tests cases that
# shows three verticals and two diagonals. As you can see, the angle between
# the vertical and the following diagonal increases from left to right.

# +---------------------------------------------+
# | *           *                    *          |
# | **          **                   *          |
# | * *         * *                  *          |
# | * *         *  *                 *          |
# | *  *        *   *                *          |
# | *  *        *    *               *          |
# | *   *       *     *              *          |
# | *   *       *      *             *          |
# | *    *      *       *            *          |
# | *    *      *        *           *          |
# | *     *     *         *          *          |
# | *     *     *          *         *          |
# | *      *    *           *        *          |
# | *      *    *            *       *          |
# | *       *   *             *      *          |
# | *       *   *              *     *          |
# | *        *  *               *    *          |
# | *        *  *                *   *          |
# | *         * *                 *  *          |
# | *         * *                  * *          |
# | *          **                   **          |
# | *           *                    *          |
# +---------------------------------------------+
#
# The big difference between 005 and 006 is that in 005 each line is individual
# line whereas for 006 it is a multi-segment line, and each vertex is listed
# only a single time. Datashader then "connects the dots" as it were.
#
# Test 007 tests the edge case, where we draw an almost staright line between
# corners with only a single pixel offset. This is to ensure that anti-aliasing
# does not try to draw pixels that  are out of bounds. Importantly, this needs
# to be run with Numba disabled, since Numba does not do OOB checking by
# default.
#
# +---------------------------------------------+
# | *                     **********************|
# |***********************                     *|
# | *                                          *|
# | *                                          *|
# | *                                          *|
# | *                                          *|
# | *                                          *|
# | *                                          *|
# | *                                          *|
# | *                                          *|
# | *                                          *|
# |*                                          * |
# |*                                          * |
# |*                                          * |
# |*                                          * |
# |*                                          * |
# |*                                          * |
# |*                                          * |
# |*                                          * |
# |*                                          * |
# |*                       *********************|
# |************************                   * |
# +---------------------------------------------+
#
#
# So for each of these 7 patterns, we test a regular and a clipped version
# (the canvas is clipped to a region in the center) in both the normal and the
# anti-aliased drawing mode. This ensures that lines can be drawn in all
# directions and that clipping a canvas works. Tests 005 and 006 ensure that
# multi-line segments can be drawn correctly too.
#
# Each tests case comes with an image and a saved xarray file (serialized in
# the NetCDF format). The image is to aid visual inspection of the algorithm
# and the quality of the anti-aliasing. The serialized xarray is for automated
# testing. The __name__ == '__main__'  section can be used to re-generate both
# the NetCDF files and the PNG files.


import os

import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest


cm = pytest.importorskip('matplotlib.cm')
binary = cm.binary

# Data directory for saving and loading test data
datadir = os.path.join(os.path.dirname(__file__), 'data')

# The colormap used for rendering
cmap01 =([tuple(v*255. for v in tuple(rgb)[:3])
          for rgb in binary(np.linspace(0, 1))])

# The normal sized canvas
regular_cvs = ds.Canvas(plot_width=50, plot_height=50,
                        x_range=(0, 49), y_range=(0, 49))

# The reduced or clipped canvas
reduced_cvs = ds.Canvas(plot_width=20, plot_height=20,
                        x_range=(14, 34), y_range=(14, 34))

# Datashader options for anti-alias and canvas size
antialias_options = ((True, "antialias"), (False, "noaa"))
canvas_options = ((regular_cvs, "normal"), (reduced_cvs, "clipped"))


def draw_line(cvs, p1, p2, antialias):
    """Draw a single line.

    Parameters
    ----------
    cvs: canvas
      A Datashader canvas
    p1: tuple
      The first vertex of the line
    p2: tuple
      The second vertex of the line
    antialias: boolean
      To anti-alias or not is the question

    Returns
    -------
    agg: A Datashader aggregator (xarray)

    """
    xs, ys = np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]])
    points = pd.DataFrame({'x': xs, 'y': ys, 'val': 5.0})
    return cvs.line(points, 'x', 'y', agg=ds.reductions.max("val"),
                    antialias=antialias)


def draw_lines(cvs, points, antialias):
    """Draw multiple line.

    Parameters
    ----------
    cvs: canvas
      A Datashader canvas
    points: list of tuple of tuple
      The lines to render as a list of tuples, where each tuple represents a
      line consisting of two tuples each containing two scalars describing the
      two vertices of the line.
    antialias: boolean
      To anti-alias or not is the question

    Returns
    -------
    agg: A Datashader aggregator (xarray)
    """
    aggs = []
    for ((x1, y1), (x2, y2)) in points:
        aggs.append(draw_line(cvs, (x1, y1), (x2, y2), antialias))
    return xr.concat(aggs, 'stack').sum(dim='stack')


def draw_multi_segment_line(cvs, points, antialias):
    """Draw multi-line segment line.

    Parameters
    ----------
    cvs: canvas
      A Datashader canvas
    points: list of tuples
      List of tuples of two scalars that represent each of the vertices in the
      multi-segment line.
    antialias: boolean
      To anti-alias or not is the question

    Returns
    -------
    agg: A Datashader aggregator (xarray)
    """
    x, y = [], []
    for (x1, y1) in points:
        x.append(x1)
        y.append(y1)
    xs, ys = np.array(x), np.array(y)
    points = pd.DataFrame({'x': xs, 'y': ys, 'val': 5.0})
    agg = cvs.line(points, 'x', 'y', agg=ds.reductions.max("val"),
                    antialias=antialias)
    # This is required for the line to render properly
    return xr.concat([agg], 'stack').sum(dim='stack')


def shade(aggregators, cmap=cmap01):
    """Shade/render the aggregator.

    Parameters
    ----------
    aggregators: xarray
      The aggregator(s)  to shade
    cmap: color map
      The colormap to use

    Returns
    -------
    img: xarray
      The shaded image.
    """
    img = ds.transfer_functions.shade(aggregators, cmap=cmap)
    img = ds.transfer_functions.set_background(img, '#ffffff')
    return img

def save_to_image(img, filename):
    """Save a shaded image as PNG file.

    Parameters
    ----------
    img: xarray
      The image to save
    filename: unicode
      The name of the file to save to, 'png' extension will be appended.

    """
    filename = os.path.join(datadir, filename + '.png')
    print('Saving: ' + filename)
    img.to_pil().save(filename)

def save_to_netcdf(img, filename):
    """Save a shaded image as NetCDF file.

    Parameters
    ----------
    img: xarray
      The image to save
    filename: unicode
      The name of the file to save to, 'nc' extension will be appended.

    """
    filename = os.path.join(datadir, filename + '.nc')
    print('Saving: ' + filename)
    img.to_netcdf(filename)


def load_from_netcdf(filename):
    """Load a shaded image from NetCDF file.

    Parameters
    ----------
    filename: unicode
      The name of the file to load from.

    Returns
    -------
    img: xarray
      The loaded image.

    """
    filename = os.path.join(datadir, filename + '.nc')
    return xr.open_dataarray(filename)

# All test generators return a 'points' list and the name of the point set

def generate_test_001():
    points = []
    for a in range(1, 55, 6):
        points.append(((1, 1), (49, a)))
    return points, "test_001"

def generate_test_002():
    points = []
    for a in range(1, 55, 6):
        points.append(((1, 1), (a, 49)))
    return points, "test_002"

def generate_test_003():
    points = []
    for a in range(1, 55, 6):
        points.append(((49, 49), (1, a)))
    return points, "test_003"

def generate_test_004():
    points = []
    for a in range(1, 55, 6):
        points.append(((49, 49), (a, 1)))
    return points, "test_004"

def generate_test_005():
    points = [
        ((1, 1),  (1, 49)),
        ((1, 49), (2, 1)),
        ((2, 1),  (2, 49)),
        ((2, 49), (4, 1)),
        ((4, 1),  (4, 49)),
        ((4, 49), (8, 1)),
        ((8, 1),  (8, 49)),
        ((8, 49), (16, 1)),
        ((16, 1), (16, 49)),
        ((16, 49),(32, 1)),
        ((32, 1), (32, 49)),
        ((32, 49),(49, 1)),
        ((49, 1), (49, 49)),
    ]
    return points, "test_005"

def generate_test_006():
    points = [
        (1, 1),
        (1, 49),
        (2, 1),
        (2, 49),
        (4, 1),
        (4, 49),
        (8, 1),
        (8, 49),
        (16, 1),
        (16, 49),
        (32, 1),
        (32, 49),
        (49, 1),
        (49, 49),
    ]
    return points, "test_006"

def generate_test_007():
    points = [
        ((0.5, 0.5),  (1.5, 48.5)),
        ((0.5, 0.5),  (48.5, 1.5)),
        ((48.5, 48.5),  (47.5, 0.5)),
        ((48.5, 48.5),  (0.5, 47.5)),
    ]
    return points, "test_007"

def generate_test_images():
    """Generate all test images.

    Returns
    -------
    results: dict
      A dictionary mapping test case name to xarray images.
    """
    results = {}
    for antialias, aa_descriptor  in antialias_options:
        for canvas, canvas_descriptor in canvas_options:
            for func in (generate_test_001,
                         generate_test_002,
                         generate_test_003,
                         generate_test_004,
                         generate_test_005,
                         generate_test_007,
                        ):
                points, name = func()
                aggregators = draw_lines(canvas, points, antialias)
                img = shade(aggregators, cmap=cmap01)
                description = "{}_{}_{}".format(
                    name, aa_descriptor, canvas_descriptor)
                results[description] = img

            for func in (generate_test_006, ):
                points, name = func()
                aggregator = draw_multi_segment_line(canvas, points, antialias)
                img = shade(aggregator, cmap=cmap01)
                description = "{}_{}_{}".format(
                    name, aa_descriptor, canvas_descriptor)
                results[description] = img
    return results

def save_test_images(images):
    """Save all images as  PNG and NetCDF files

    Parameters
    ----------
    images: dict
      A dictionary mapping test case names to xarray images.
    """
    for description, img in images.items():
        save_to_image(img, description)
        save_to_netcdf(img, description)

def load_test_images(images):
    """Load all images from NetCDF files

    Returns
    -------
    loaded: dict
      A dictionary mapping test case names to xarray images.
    """
    loaded = {}
    for description, _ in images.items():
        loaded[description] = load_from_netcdf(description)
    return loaded

def test_antialiasing():
    """Test case for all images.

    Will generate all test cases, then load all test cases from disk and
    compare them with each other

    """
    images  = generate_test_images()
    loaded = load_test_images(images)
    print(list(loaded.keys()))
    for description in images.keys():
        assert (images[description] == loaded[description]).all()

if __name__ == '__main__':
    # Run this to generate the PNG and NetCDF files.
    images  = generate_test_images()
    print(list(images.keys()))
    save_test_images(images)
