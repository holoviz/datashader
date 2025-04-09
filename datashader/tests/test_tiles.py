from __future__ import annotations

import pytest

import datashader as ds
import datashader.transfer_functions as tf

from datashader.colors import viridis

from datashader.tiles import render_tiles
from datashader.tiles import gen_super_tiles
from datashader.tiles import _get_super_tile_min_max
from datashader.tiles import calculate_zoom_level_stats
from datashader.tiles import MercatorTileDefinition

from datashader.tests.utils import dask_skip

import numpy as np
import pandas as pd

TOLERANCE = 0.01

MERCATOR_CONST = 20037508.34

df = None
def mock_load_data_func(x_range, y_range):
    global df
    if df is None:
        rng = np.random.default_rng()
        xs = rng.normal(loc=0, scale=500000, size=10000000)
        ys = rng.normal(loc=0, scale=500000, size=10000000)
        df = pd.DataFrame(dict(x=xs, y=ys))

    return df.loc[df['x'].between(*x_range) & df['y'].between(*y_range)]


def mock_rasterize_func(df, x_range, y_range, height, width):
    cvs = ds.Canvas(x_range=x_range, y_range=y_range,
                    plot_height=height, plot_width=width)
    agg = cvs.points(df, 'x', 'y')
    return agg


def mock_shader_func(agg, span=None):
    img = tf.shade(agg, cmap=viridis, span=span, how='log')
    img = tf.set_background(img, 'black')
    return img


def mock_post_render_func(img, **kwargs):
    ImageDraw = pytest.importorskip("PIL.ImageDraw")

    (x, y) = (5, 5)
    info = "x={} / y={} / z={}, w={}, h={}".format(kwargs['x'],
                                                   kwargs['y'],
                                                   kwargs['z'],
                                                   img.width,
                                                   img.height)

    draw = ImageDraw.Draw(img)
    draw.text((x, y), info, fill='rgb(255, 255, 255)')
    return img


# TODO: mark with slow_test
@dask_skip
def test_render_tiles():
    pytest.importorskip("PIL")

    full_extent_of_data = (-500000, -500000,
                           500000, 500000)
    levels = list(range(2))
    output_path = 'test_tiles_output'
    results = render_tiles(full_extent_of_data,
                           levels,
                           load_data_func=mock_load_data_func,
                           rasterize_func=mock_rasterize_func,
                           shader_func=mock_shader_func,
                           post_render_func=mock_post_render_func,
                           output_path=output_path)

    assert results
    assert isinstance(results, dict)

    for level in levels:
        assert level in results
        assert isinstance(results[level], dict)

    assert results[0]['success']
    assert results[0]['stats']
    assert results[0]['supertile_count']


def assert_is_numeric(value):
    is_int_or_float = isinstance(value, (int, float))
    type_name = type(value).__name__
    is_numpy_int_or_float = 'int' in type_name or 'float' in type_name
    assert any([is_int_or_float, is_numpy_int_or_float])



def test_get_super_tile_min_max():

    tile_info = {'level': 0,
                'x_range': (-MERCATOR_CONST, MERCATOR_CONST),
                'y_range': (-MERCATOR_CONST, MERCATOR_CONST),
                'tile_size': 256,
                'span': (0, 1000)}

    agg = _get_super_tile_min_max(tile_info, mock_load_data_func, mock_rasterize_func)

    result = [np.nanmin(agg.data), np.nanmax(agg.data)]

    assert isinstance(result, list)
    assert len(result) == 2
    assert_is_numeric(result[0])
    assert_is_numeric(result[1])

@dask_skip
def test_calculate_zoom_level_stats_with_fullscan_ranging_strategy():
    full_extent = (-MERCATOR_CONST, -MERCATOR_CONST,
                   MERCATOR_CONST, MERCATOR_CONST)
    level = 0
    color_ranging_strategy = 'fullscan'
    super_tiles, span = calculate_zoom_level_stats(list(gen_super_tiles(full_extent, level)),
                                        mock_load_data_func,
                                        mock_rasterize_func,
                                        color_ranging_strategy=color_ranging_strategy)

    assert isinstance(span, (list, tuple))
    assert len(span) == 2
    assert_is_numeric(span[0])
    assert_is_numeric(span[1])

def test_meters_to_tile():
    # Part of NYC (used in taxi demo)
    full_extent_of_data = (-8243206.93436, 4968192.04221, -8226510.539480001, 4982886.20438)
    xmin, ymin, xmax, ymax = full_extent_of_data
    zoom = 12
    tile_def = MercatorTileDefinition((xmin, xmax), (ymin, ymax), tile_size=256)
    tile = tile_def.meters_to_tile(xmin, ymin, zoom)
    assert tile == (1205, 1540) # using Google tile coordinates, not TMS
