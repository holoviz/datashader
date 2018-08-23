import datashader as ds
import datashader.transfer_functions as tf

from datashader.colors import viridis

from datashader.tiles import render_tiles
from datashader.tiles import _get_super_tile_min_max
from datashader.tiles import calculate_zoom_level_stats

import numpy as np
import pandas as pd

TOLERANCE = 0.01

MERCATOR_CONST = 20037508.34

df = None
def mock_load_data_func(x_range, y_range):
    global df
    if df is None:
        xs = np.random.normal(loc=0, scale=500000, size=10000000)
        ys = np.random.normal(loc=0, scale=500000, size=10000000)
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


def mock_post_render_func(img, extras=None):
    from PIL import ImageDraw

    (x, y) = (5, 5)
    info = "x={} / y={} / z={}, w={}, h={}".format(extras['x'],
                                                   extras['y'],
                                                   extras['z'],
                                                   img.width,
                                                   img.height)

    draw = ImageDraw.Draw(img)
    draw.text((x, y), info, fill='rgb(255, 255, 255)')
    return img


# TODO: mark with slow_test
def test_render_tiles():
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

    # TODO: assert more!
    assert results
    assert isinstance(results, dict)

    for l in levels:
        assert l in results
        assert isinstance(results[l], dict)

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

    result = _get_super_tile_min_max(tile_info, mock_load_data_func, mock_rasterize_func)

    assert isinstance(result, list)
    assert len(result) == 2
    assert_is_numeric(result[0])
    assert_is_numeric(result[1])

def test_calculate_zoom_level_stats_with_fullscan_ranging_strategy():
    full_extent = (-MERCATOR_CONST, -MERCATOR_CONST,
                   MERCATOR_CONST, MERCATOR_CONST)
    level = 0
    color_ranging_strategy = 'fullscan'
    result = calculate_zoom_level_stats(full_extent, level,
                                        mock_load_data_func,
                                        mock_rasterize_func,
                                        color_ranging_strategy=color_ranging_strategy)

    assert isinstance(result, (list, tuple))
    assert len(result) == 2
    assert_is_numeric(result[0])
    assert_is_numeric(result[1])

'''

def test_gen_super_tiles():
    assert False


def test_render_super_tile():
    assert False


def test_MercatorTileDefinition():
    assert False


def test_to_ogc_tile_metadata():
    assert False


def test_to_esri_tile_metadata():
    assert False


def test_get_resolution_by_extent():
    assert False


def test_get_level_by_extent():
    assert False


def test_pixels_to_meters():
    assert False


def test_meters_to_pixels():
    assert False


def test_pixels_to_tile():
    assert False


def test_pixels_to_raster():
    assert False


def test_meters_to_tile():
    assert False


def test_get_tiles_by_extent():
    assert False


def test_get_tile_meters():
    assert False


def test_can_render_tiles_to_filesystem():
    assert False


def test_can_render_tiles_to_s3():
    assert False

def test_should_get_tiles_for_extent_correctly():
    url = 'http://c.tiles.mapbox.com/v3/examples.map-szwdot65/{Z}/{X}/{Y}.png'
    source = TMSTileSource({url: url})

    assert T.expect_mercator_tile_counts(source)


def test_should_successfully_set_x_offset_and_y_offset():
    tile_options = {
        x_origin_offset: 0,

        y_origin_offset: 0
    }
    offset_source = TMSTileSource(tile_options)

    assert offset_source.x_origin_offse == 0
    assert offset_source.y_origin_offse == 0


def test_should_account_x_offset_and_y_offset():
    tile_options = {
        x_origin_offset: 0,

        y_origin_offset: 0
    }
    offset_source = TMSTileSource(tile_options)
    bounds = offset_source.get_tile_meter_bounds(0, 0, 16)
    assert 0 in bounds


def test_should_calculate_resolution():
    assert source.get_resolution(1) == pytest.approx(78271.517, TOLERANCE)
    assert source.get_resolution(12) == pytest.approx(38.2185, TOLERANCE)


def test_should_get_tiles_for_extent_correctly_2():
    tile_options = {
        url: 'http://mt0.google.com/vt/lyrs=m@169000000&hl=en&x={X}&y={Y}&z={Z}&s=Ga'

    }

    source = WMTSTileSource(tile_options)

    assert T.expect_mercator_tile_counts(source)


def test_should_get_tile_bounds_in_meters():
    x, y, z = source.wmts_to_tms(511, 845, 11)
    bounds = source.get_tile_meter_bounds(x, y, z)
    assert bounds[0] == pytest.approx(-10038322.050635627, TOLERANCE)
    assert bounds[1] == pytest.approx(3483082.504898913, TOLERANCE)
    assert bounds[2] == pytest.approx(-10018754.171394622, TOLERANCE)
    assert bounds[3] == pytest.approx(3502650.384139918, TOLERANCE)


def test_should_get_tile_bounds_in_lat_lang():
    x, y, z = source.wmts_to_tms(511, 845, 11)
    bounds = source.get_tile_meter_bounds(x, y, z)
    assert bounds[0] == pytest.approx(-90.17578125, TOLERANCE)
    assert bounds[1] == pytest.approx(29.840643899834436, TOLERANCE)
    assert bounds[2] == pytest.approx(-90, TOLERANCE)
    assert bounds[3] == pytest.approx(29.99300228455108, TOLERANCE)


def test_should_get_tiles_for_extent_correctly_3():
    tile_options = {
        url: 'http://t0.tiles.virtualearth.net/tiles/a{Q}.jpeg?g=854&mkt=en-US&token=Anz84uRE1RULeLwuJ0qKu5amcu5rugRXy1vKc27wUaKVyIv1SVZrUjqaOfXJJoI0'

    }

    source = QUADKEYTileSource(tile_options)

    assert T.expect_mercator_tile_counts(source)


def test_should_convert_tile_xyz_to_quadkey():
    assert source.tile_xyz_to_quadkey(0, 0, 0) == ''
    assert source.tile_xyz_to_quadkey(0, 0, 1) == '0'
    assert source.tile_xyz_to_quadkey(0, 0, 2) == '00'
    assert source.tile_xyz_to_quadkey(20, 30, 10) == '0000032320'


def test_should_convert_tile_quadkey_to_xyz():
    assert source.quadkey_to_tile_xyz('') == [0, 0, 0]
    assert source.quadkey_to_tile_xyz('0') == [0, 0, 1]
    assert source.quadkey_to_tile_xyz('00') == [0, 0, 2]
    assert source.quadkey_to_tile_xyz('0000032320') == [20, 30, 10]


def test_should_get_tiles_for_extent_correctly_4():
    tile_options = {
        url: 'http://maps.ngdc.noaa.gov/soap/web_mercator/dem_hillshades/MapServer/WMSServer?request=GetMap&service=WMS&styles=default&version=1.3.0&format=image/png&bbox={XMIN},{YMIN},{XMAX},{YMAX}&width=256&height=256&crs=3857&layers=DEM%20Hillshades&BGCOLOR=0x000000&transparent=true'
    }

    source = BBoxTileSource(tile_options)

    assert T.expect_mercator_tile_counts(source)


def test_should_handle_case_insensitive_url_parameters_2():
    tile_options = {
        url: 'http://maps.ngdc.noaa.gov/soap/web_mercator/dem_hillshades/MapServer/WMSServer?request=GetMap&service=WMS&styles=default&version=1.3.0&format=image/png&bbox={XMIN},{YMIN},{XMAX},{YMAX}&width=256&height=256&crs=3857&layers=DEM%20Hillshades&BGCOLOR=0x000000&transparent=true'
    }
    tile_source = BBoxTileSource(tile_options)
    url = tile_source.get_image_url(0, 0, 0)
    assert url.indexOf('{xmin}') == -1
    assert url.indexOf('{ymin}') == -1
    assert url.indexOf('{xmax}') == -1
    assert url.indexOf('{ymax}') == -1
    tile_options.url = 'http://mock?bbox={XMIN},{YMIN},{XMAX},{YMAX}'
    tile_source = BBoxTileSource(tile_options)
    assert url.indexOf('{XMIN}') == -1
    assert url.indexOf('{YMIN}') == -1
    assert url.indexOf('{XMAX}') == -1
    assert url.indexOf('{YMAX}') == -1


source = MercatorTileSource()


def test_should_calculate_resolution_2():
    assert source.get_resolution(1) == pytest.approx(78271.517, TOLERANCE)
    assert source.get_resolution(12) == pytest.approx(38.2185, TOLERANCE)


def test_should_convert_tile_xyz_into_cache_key():
    assert source.tile_xyz_to_key(1, 1, 1) == "1:1:1"


def test_should_convert_tile_cache_key_into_xyz():
    assert source.key_to_tile_xyz("1:1:1") == [1, 1, 1]


def test_should_successfully_wrap_around_x_for_normalized_tile_coordinates():
    assert source.normalize_xyz(-1, 1, 2) == [3, 1, 2]


def test_should_successfully_get_closest_parent_tile_by_xyz():
    source.tiles[source.tile_xyz_to_key(0, 1, 1)] = {}

    assert source.get_closest_parent_by_tile_xyz(0, 3, 2) == [0, 1, 1]


def test_should_verify_whether_tile_xyz_are_valid():
    tile_options = {
        wrap_around: True
    }

    source = MercatorTileSource(tile_options)
    assert source.is_valid_tile(-1, 1, 1) == True

    tile_options = {
        wrap_around: False
    }

    source = MercatorTileSource(tile_options)

    assert source.is_valid_tile(-1, 1, 1) == False


def test_should_not_snap_to_zoom_level():
    bounds = source.snap_to_zoom_level(T.MERCATOR_BOUNDS, 400, 400, 2)

    assert bounds[0] == pytest.approx(T.MERCATOR_BOUNDS[0], TOLERANCE)
    assert bounds[1] == pytest.approx(T.MERCATOR_BOUNDS[1], TOLERANCE)
    assert bounds[2] == pytest.approx(T.MERCATOR_BOUNDS[2], TOLERANCE)
    assert bounds[3] == pytest.approx(T.MERCATOR_BOUNDS[3], TOLERANCE)


def test_should_get_best_zoom_level_based_on_extent_and_height_width():
    assert source.get_level_by_extent(T.MERCATOR_BOUNDS, 256, 256) == 0
    assert source.get_level_by_extent(T.MERCATOR_BOUNDS, 512, 512) == 1
    assert source.get_level_by_extent(T.MERCATOR_BOUNDS, 1024, 1024) == 2


def test_should_last_zoom_level_as_best_when_there_are_no_others():
    assert source.get_level_by_extent(T.MERCATOR_BOUNDS, 1e40, 1e40) == 30


def test_should_get_best_zoom_level_based_on_extent_and_height_width_2():
    assert source.get_closest_level_by_extent(T.MERCATOR_BOUNDS, 256, 256) == 0
    assert source.get_closest_level_by_extent(T.MERCATOR_BOUNDS, 512, 512) == 1
    assert source.get_closest_level_by_extent(T.MERCATOR_BOUNDS, 1024, 1024) == 2


def test_conpixel_xy_to_tile_xy_vert():
    assert source.pixels_to_tile(1, 1) == [0, 0]
    assert source.pixels_to_tile(0, 0) == [0, 0]


def test_convert_pixel_xy_to_meters_xy():
    assert source.pixels_to_meters(0, 0, 0) == [-20037508.34, -20037508.34]


def test_should_get_tile_bounds_in_meters_2():
    bounds = source.get_tile_meter_bounds(511, 1202, 11)

    assert bounds[0] == pytest.approx(-10038322.050635627, TOLERANCE)
    assert bounds[1] == pytest.approx(3483082.504898913, TOLERANCE)
    assert bounds[2] == pytest.approx(-10018754.171394622, TOLERANCE)
    assert bounds[3] == pytest.approx(3502650.384139918, TOLERANCE)


def test_should_get_tile_bounds_in_lat_lng():
    bounds = source.get_tile_geographic_bounds(511, 1202, 11)

    assert bounds[0] == pytest.approx(-90.17578125, TOLERANCE)
    assert bounds[1] == pytest.approx(29.840643899834436, TOLERANCE)
    assert bounds[2] == pytest.approx(-90, TOLERANCE)
    assert bounds[3] == pytest.approx(29.99300228455108, TOLERANCE)


def test_should_get_tile_urls_by_geographic_extent():
    tile_options = {
        url: 'http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png'
    }
    tile_source = TMSTileSource(tile_options)
    xmin, ymin, xmax, ymax, level = [-90.283741, 29.890626, -89.912952,
                                     30.057766, 11]
    expected_tiles = []
    expected_tiles.append('http://c.tile.openstreetmap.org/11/510/1201.png')
    expected_tiles.append('http://c.tile.openstreetmap.org/11/511/1201.png')
    expected_tiles.append('http://c.tile.openstreetmap.org/11/512/1201.png')
    expected_tiles.append('http://c.tile.openstreetmap.org/11/510/1202.png')
    expected_tiles.append('http://c.tile.openstreetmap.org/11/511/1202.png')
    expected_tiles.append('http://c.tile.openstreetmap.org/11/512/1202.png')
    urls = source.get_tiles_by_extent(xmin, ymin, xmax, ymax, level)
    for url in expected_tiles:
        assert expected_tiles.index(url) < -1

'''
