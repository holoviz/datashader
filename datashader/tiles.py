from __future__ import absolute_import, division, print_function

import math
import os
import sqlite3
import uuid
from io import BytesIO

import dask.bag as db
import numpy as np
import xarray
from PIL.Image import fromarray

from .utils import meters_to_lnglat

try:
    import netCDF4
except Exception:
    netCDF4 = None

__all__ = ['render_tiles', 'MercatorTileDefinition']


# helpers ---------------------------------------------------------------------
def _create_dir(path):
    import os, errno

    try:
        if os.path.isdir(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _get_super_tile_min_max(tile_info, load_data_func, rasterize_func, local_cache_path=None):
    df = load_data_func(tile_info['x_range'], tile_info['y_range'])
    agg = rasterize_func(df, x_range=tile_info['x_range'],
                         y_range=tile_info['y_range'],
                         height=tile_info['tile_size'], width=tile_info['tile_size'])
    tile_info['agg_type'] = agg.dtype.kind
    tile_info['span_min'] = np.nanmin(agg.data)
    tile_info['span_max'] = np.nanmax(agg.data)

    if local_cache_path:
        cache_file_path = os.path.join(local_cache_path, 'super_tile_' + str(uuid.uuid4()) + '.nc')
        agg.to_netcdf(cache_file_path, engine='netcdf4', format='NETCDF4')
        tile_info['cache_file'] = cache_file_path

    return tile_info


def calculate_zoom_level_stats(super_tiles, load_data_func,
                               rasterize_func,
                               color_ranging_strategy='fullscan', local_cache_path=None):
    if color_ranging_strategy == 'fullscan':

        super_tiles = db.from_sequence(super_tiles).map(_get_super_tile_min_max, load_data_func,
                                                        rasterize_func, local_cache_path).compute()

        span_min = None
        span_max = None
        is_bool = False

        for super_tile in super_tiles:
            if super_tile['agg_type'] == 'b':
                is_bool = True
            else:
                if span_min is None:
                    span_min = super_tile['span_min']
                else:
                    span_min = min(span_min, super_tile['span_min'])

                if span_max is None:
                    span_max = super_tile['span_max']
                else:
                    span_max = max(span_max, super_tile['span_max'])

        if is_bool:
            span = (0, 1)
        else:
            span = (span_min, span_max)
        return super_tiles, span
    else:
        raise ValueError('Invalid color_ranging_strategy option')


def render_tiles(full_extent, levels, load_data_func,
                 rasterize_func, shader_func,
                 post_render_func, output_path, color_ranging_strategy='fullscan',
                 local_cache_path=None):
    results = dict()

    _setup(full_extent, levels, output_path, local_cache_path)

    for level in levels:
        print('calculating statistics for level {}'.format(level))
        super_tiles, span = calculate_zoom_level_stats(list(gen_super_tiles(full_extent, level)),
                                                       load_data_func, rasterize_func,
                                                       color_ranging_strategy=color_ranging_strategy,
                                                       local_cache_path=local_cache_path)
        print('rendering {} supertiles for zoom level {} with span={}'.format(len(super_tiles), level, span))
        b = db.from_sequence(super_tiles)
        b.map(render_super_tile, span, output_path, load_data_func, rasterize_func, shader_func, post_render_func,
              local_cache_path).compute()
        results[level] = dict(success=True, stats=span, supertile_count=len(super_tiles))

    return results


def gen_super_tiles(extent, zoom_level, span=None):
    xmin, ymin, xmax, ymax = extent
    super_tile_size = min(2 ** 4 * 256,
                          (2 ** zoom_level) * 256)
    super_tile_def = MercatorTileDefinition(x_range=(xmin, xmax), y_range=(ymin, ymax), tile_size=super_tile_size)

    for s in super_tile_def.get_tiles_by_extent(extent, zoom_level):
        st_extent = s[3]
        x_range = (st_extent[0], st_extent[2])
        y_range = (st_extent[1], st_extent[3])
        yield {'level': zoom_level,
               'x_range': x_range,
               'y_range': y_range,
               'tile_size': super_tile_def.tile_size,
               'span': span}


def render_super_tile(tile_info, span, output_path, load_data_func, rasterize_func, shader_func, post_render_func,
                      local_cache_path):
    agg = None

    if local_cache_path is not None:
        agg = xarray.load_dataarray(tile_info['cache_file'], engine='netcdf4')
        os.remove(tile_info['cache_file'])
    else:
        df = load_data_func(tile_info['x_range'], tile_info['y_range'])
        agg = rasterize_func(df, x_range=tile_info['x_range'],
                             y_range=tile_info['y_range'],
                             height=tile_info['tile_size'], width=tile_info['tile_size'])

    ds_img = shader_func(agg, span=span)

    return create_sub_tiles(ds_img, tile_info, output_path, post_render_func)


def _setup(full_extent, levels, output_path, local_cache_path):
    # validate / createoutput_dir
    _create_dir(output_path)

    if output_path.endswith("mbtiles"):
        _create_dir(os.path.dirname(output_path))
        # Create mbtiles file and setup sqlite tables.
        MapboxTileRenderer.setup(output_path, full_extent, levels[0], levels[len(levels) - 1])

    if local_cache_path:
        assert netCDF4, 'netcdf4 library must be installed for use with local_cache.'
        _create_dir(local_cache_path)


def create_sub_tiles(data_array, tile_info, output_path, post_render_func=None):
    # create tile source
    tile_def = MercatorTileDefinition(x_range=tile_info['x_range'],
                                      y_range=tile_info['y_range'],
                                      tile_size=256)

    # create Tile Renderer
    if output_path.endswith("mbtiles"):
        renderer = MapboxTileRenderer(tile_def, output_location=output_path,
                                      post_render_func=post_render_func)
    elif output_path.startswith('s3:'):
        renderer = S3TileRenderer(tile_def, output_location=output_path,
                                  post_render_func=post_render_func)
    else:
        renderer = FileSystemTileRenderer(tile_def, output_location=output_path,
                                          post_render_func=post_render_func)

    return renderer.render(data_array, level=tile_info['level'])


def invert_y_tile(y, z):
    # Convert from TMS to Google tile y coordinate, and vice versa
    return (2 ** z) - 1 - y


# TODO: change name from source to definition
class MercatorTileDefinition(object):
    ''' Implementation of mercator tile source
    In general, tile sources are used as a required input for ``TileRenderer``.

    Parameters
    ----------

    x_range : tuple
      full extent of x dimension in data units

    y_range : tuple
      full extent of y dimension in data units

    max_zoom : int
      A maximum zoom level for the tile layer. This is the most zoomed-in level.

    min_zoom : int
      A minimum zoom level for the tile layer. This is the most zoomed-out level.

    max_zoom : int
      A maximum zoom level for the tile layer. This is the most zoomed-in level.

    x_origin_offset : int
      An x-offset in plot coordinates.

    y_origin_offset : int
      An y-offset in plot coordinates.

    initial_resolution : int
      Resolution (plot_units / pixels) of minimum zoom level of tileset
      projection. None to auto-compute.

    format : int
      An y-offset in plot coordinates.

    Output
    ------
    tileScheme: MercatorTileSource

    '''

    def __init__(self, x_range, y_range, tile_size=256, min_zoom=0, max_zoom=30,
                 x_origin_offset=20037508.34, y_origin_offset=20037508.34,
                 initial_resolution=156543.03392804097):
        self.x_range = x_range
        self.y_range = y_range
        self.tile_size = tile_size
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.x_origin_offset = x_origin_offset
        self.y_origin_offset = y_origin_offset
        self.initial_resolution = initial_resolution
        self._resolutions = [self._get_resolution(z) for z in range(self.min_zoom, self.max_zoom + 1)]

    def to_ogc_tile_metadata(self, output_file_path):
        '''
        Create OGC tile metadata XML
        '''
        pass

    def to_esri_tile_metadata(self, output_file_path):
        '''
        Create ESRI tile metadata JSON
        '''
        pass

    def is_valid_tile(self, x, y, z):

        if x < 0 or x >= math.pow(2, z):
            return False

        if y < 0 or y >= math.pow(2, z):
            return False

        return True

    # TODO ngjit?
    def _get_resolution(self, z):
        return self.initial_resolution / (2 ** z)

    def get_resolution_by_extent(self, extent, height, width):
        x_rs = (extent[2] - extent[0]) / width
        y_rs = (extent[3] - extent[1]) / height
        return [x_rs, y_rs]

    def get_level_by_extent(self, extent, height, width):
        x_rs = (extent[2] - extent[0]) / width
        y_rs = (extent[3] - extent[1]) / height
        resolution = max(x_rs, y_rs)

        # TODO: refactor this...
        i = 0
        for r in self._resolutions:
            if resolution > r:
                if i == 0:
                    return 0
                if i > 0:
                    return i - 1
            i += 1
        return (i - 1)

    def pixels_to_meters(self, px, py, level):
        res = self._get_resolution(level)
        mx = (px * res) - self.x_origin_offset
        my = (py * res) - self.y_origin_offset
        return (mx, my)

    def meters_to_pixels(self, mx, my, level):
        res = self._get_resolution(level)
        px = (mx + self.x_origin_offset) / res
        py = (my + self.y_origin_offset) / res
        return (px, py)

    def pixels_to_tile(self, px, py, level):
        tx = math.ceil(px / self.tile_size)
        tx = tx if tx == 0 else tx - 1
        ty = max(math.ceil(py / self.tile_size) - 1, 0)
        # convert from TMS y coordinate
        return (int(tx), invert_y_tile(int(ty), level))

    def pixels_to_raster(self, px, py, level):
        map_size = self.tile_size << level
        return (px, map_size - py)

    def meters_to_tile(self, mx, my, level):
        px, py = self.meters_to_pixels(mx, my, level)
        return self.pixels_to_tile(px, py, level)

    def get_tiles_by_extent(self, extent, level):

        # unpack extent and convert to tile coordinates
        xmin, ymin, xmax, ymax = extent
        # note y coordinates are reversed since they are in opposite direction to meters
        txmin, tymax = self.meters_to_tile(xmin, ymin, level)
        txmax, tymin = self.meters_to_tile(xmax, ymax, level)

        for ty in range(tymin, tymax + 1):
            for tx in range(txmin, txmax + 1):
                if self.is_valid_tile(tx, ty, level):
                    t = (tx, ty, level, self.get_tile_meters(tx, ty, level))
                    yield t

    def get_tile_meters(self, tx, ty, level):
        ty = invert_y_tile(ty, level)  # convert to TMS for conversion to meters
        xmin, ymin = self.pixels_to_meters(tx * self.tile_size, ty * self.tile_size, level)
        xmax, ymax = self.pixels_to_meters((tx + 1) * self.tile_size, (ty + 1) * self.tile_size, level)
        return (xmin, ymin, xmax, ymax)


class TileRenderer(object):

    def __init__(self, tile_definition, output_location, tile_format='PNG',
                 post_render_func=None):

        self.tile_def = tile_definition
        self.output_location = output_location
        self.tile_format = tile_format
        self.post_render_func = post_render_func

        # TODO: add all the formats supported by PIL
        if self.tile_format not in ('PNG', 'JPG'):
            raise ValueError('Invalid output format')

    def render(self, da, level):
        xmin, xmax = self.tile_def.x_range
        ymin, ymax = self.tile_def.y_range
        extent = xmin, ymin, xmax, ymax

        for t in self.tile_def.get_tiles_by_extent(extent, level):
            x, y, z, data_extent = t
            dxmin, dymin, dxmax, dymax = data_extent

            arr = da.loc[{da.dims[1]: slice(dxmin, dxmax), da.dims[0]: slice(dymin, dymax)}]

            if 0 in arr.shape:
                continue

            img = fromarray(np.flip(arr.data, 0), 'RGBA')  # flip since y tiles go down (Google map tiles)

            if self.post_render_func:
                extras = dict(x=x, y=y, z=z)
                img = self.post_render_func(img, **extras)

            yield (img, x, y, z)


def tile_previewer(full_extent, tileset_url,
                   output_dir=None,
                   filename='index.html',
                   title='Datashader Tileset',
                   min_zoom=0, max_zoom=40,
                   height=None, width=None,
                   **kwargs):
    '''Helper function for creating a simple Bokeh figure with
    a WMTS Tile Source.

    Notes
    -----
    - if you don't supply height / width, stretch_both sizing_mode is used.
    - supply an output_dir to write figure to disk.
    '''

    try:
        from bokeh.plotting import figure
        from bokeh.models.tiles import WMTSTileSource
        from bokeh.io import output_file, save
        from os import path
    except ImportError:
        raise ImportError('conda install bokeh to enable creation of simple tile viewer')

    if output_dir:
        output_file(filename=path.join(output_dir, filename),
                    title=title)

    xmin, ymin, xmax, ymax = full_extent

    if height and width:
        p = figure(width=width, height=height,
                   x_range=(xmin, xmax),
                   y_range=(ymin, ymax),
                   tools="pan,wheel_zoom,reset", **kwargs)
    else:
        p = figure(sizing_mode='stretch_both',
                   x_range=(xmin, xmax),
                   y_range=(ymin, ymax),
                   tools="pan,wheel_zoom,reset", **kwargs)

    p.background_fill_color = 'black'
    p.grid.grid_line_alpha = 0
    p.axis.visible = True

    tile_source = WMTSTileSource(url=tileset_url,
                                 min_zoom=min_zoom,
                                 max_zoom=max_zoom)
    p.add_tile(tile_source, render_parents=False)

    if output_dir:
        save(p)

    return p


class FileSystemTileRenderer(TileRenderer):

    def render(self, da, level):
        for img, x, y, z in super(FileSystemTileRenderer, self).render(da, level):
            tile_file_name = '{}.{}'.format(y, self.tile_format.lower())
            tile_directory = os.path.join(self.output_location, str(z), str(x))
            output_file = os.path.join(tile_directory, tile_file_name)
            _create_dir(tile_directory)
            img.save(output_file, self.tile_format)


class S3TileRenderer(TileRenderer):

    def render(self, da, level):

        try:
            import boto3
        except ImportError:
            raise ImportError('conda install boto3 to enable rendering to S3')

        try:
            from urlparse import urlparse
        except ImportError:
            from urllib.parse import urlparse

        s3_info = urlparse(self.output_location)
        bucket = s3_info.netloc
        client = boto3.client('s3')
        for img, x, y, z in super(S3TileRenderer, self).render(da, level):
            tile_file_name = '{}.{}'.format(y, self.tile_format.lower())
            key = os.path.join(s3_info.path, str(z), str(x), tile_file_name).lstrip('/')
            output_buf = BytesIO()
            img.save(output_buf, self.tile_format)
            output_buf.seek(0)
            client.put_object(Body=output_buf, Bucket=bucket, Key=key, ACL='public-read')

        return 'https://{}.s3.amazonaws.com/{}'.format(bucket, s3_info.path)


class MapboxTileRenderer(TileRenderer):

    @staticmethod
    def setup(output_location, full_extent, min_zoom, max_zoom, tile_format="PNG"):
        con = sqlite3.connect(output_location)
        cur = con.cursor()

        # Create MBTiles tables.
        cur.execute("""
            create table if not exists tiles (
                zoom_level integer,
                tile_column integer,
                tile_row integer,
                tile_data blob);
                """)
        cur.execute("""create table if not exists metadata
            (name text, value text);""")
        cur.execute("""create unique index if not exists name on metadata (name);""")
        cur.execute("""create unique index if not exists tile_index on tiles
            (zoom_level, tile_column, tile_row);""")

        # Compute Extents in WGS84
        min_lon, min_lat = meters_to_lnglat(full_extent[0], full_extent[1])
        max_lon, max_lat = meters_to_lnglat(full_extent[2], full_extent[3])

        # Compute the Center & Zoom Level
        lat_diff = max_lat - min_lat
        lon_diff = max_lon - min_lon

        lat_center = min_lat + (lat_diff / 2)
        lon_center = min_lon + (lon_diff / 2)

        max_diff = max(lon_diff, lat_diff)
        zoom_level = None
        if max_diff < (360.0 / math.pow(2, 20)):
            zoom_level = 21
        else:
            zoom_level = int(-1 * ((math.log(max_diff) / math.log(2.0)) - (math.log(360.0) / math.log(2))))
            if zoom_level < 1:
                zoom_level = 1

        # Setup MBTiles metadata table.
        cur.execute("""insert or replace into metadata (name, value) values (?, ?);""",
                    ("name", os.path.splitext(os.path.basename(output_location))[0]))
        cur.execute("""insert or replace into metadata (name, value) values (?, ?);""",
                    ("format", tile_format.lower()))
        cur.execute("""insert or replace into metadata (name, value) values (?, ?);""",
                    ("bounds", "{},{},{},{}".format(min_lon, min_lat, max_lon, max_lat)))
        cur.execute("""insert or replace into metadata (name, value) values (?, ?);""",
                    ("center", "{},{},{}".format(lon_center, lat_center, zoom_level)))
        cur.execute("""insert or replace into metadata (name, value) values (?, ?);""",
                    ("minzoom", min_zoom))
        cur.execute("""insert or replace into metadata (name, value) values (?, ?);""",
                    ("maxzoom", max_zoom))

        cur.close()
        con.commit()
        con.close()

    def render(self, da, level):
        con = sqlite3.connect(self.output_location, isolation_level=None)
        cur = con.cursor()

        for img, x, y, z in super(MapboxTileRenderer, self).render(da, level):
            image_bytes = BytesIO()
            img.save(image_bytes, self.tile_format)
            image_bytes.seek(0)

            tile_row = (2 ** z) - 1 - y

            cur.execute("""insert or replace into tiles (zoom_level,
                tile_column, tile_row, tile_data) values
                (?, ?, ?, ?);""",
                        (z, x, tile_row, sqlite3.Binary(image_bytes.getvalue())))

        cur.close()
        con.commit()
        con.close()
