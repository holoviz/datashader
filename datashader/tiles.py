from __future__ import annotations
from importlib.util import find_spec
from io import BytesIO

import math
import os


import numpy as np

try:
    import dask
    import dask.bag as db
except ImportError:
    dask, db = None, None

__all__ = ['render_tiles', 'MercatorTileDefinition']


# helpers ---------------------------------------------------------------------
def _create_dir(path):
    import errno
    import os

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _get_super_tile_min_max(tile_info, load_data_func, rasterize_func):
    tile_size = tile_info['tile_size']
    df = load_data_func(tile_info['x_range'], tile_info['y_range'])
    agg = rasterize_func(df, x_range=tile_info['x_range'],
                         y_range=tile_info['y_range'],
                         height=tile_size, width=tile_size)
    return agg


def calculate_zoom_level_stats(super_tiles, load_data_func,
                               rasterize_func,
                               color_ranging_strategy='fullscan'):
    if color_ranging_strategy == 'fullscan':
        stats = []
        is_bool = False
        for super_tile in super_tiles:
            agg = _get_super_tile_min_max(super_tile, load_data_func, rasterize_func)
            super_tile['agg'] = agg
            if agg.dtype.kind == 'b':
                is_bool = True
            else:
                stats.append(np.nanmin(agg.data))
                stats.append(np.nanmax(agg.data))
        if is_bool:
            span = (0, 1)
        elif dask:
            b = db.from_sequence(stats)
            span = dask.compute(b.min(), b.max())
        else:
            raise ValueError('Dask is required for non-boolean data')
        return super_tiles, span
    else:
        raise ValueError('Invalid color_ranging_strategy option')


def render_tiles(full_extent, levels, load_data_func,
                 rasterize_func, shader_func,
                 post_render_func, output_path, color_ranging_strategy='fullscan'):
    if not dask:
        raise ImportError('Dask is required for rendering tiles')
    results = {}
    for level in levels:
        print('calculating statistics for level {}'.format(level))
        super_tiles, span = calculate_zoom_level_stats(list(gen_super_tiles(full_extent, level)),
                                                       load_data_func, rasterize_func,
                                                       color_ranging_strategy=color_ranging_strategy)
        print('rendering {} supertiles for zoom level {} with span={}'.format(len(super_tiles),
                                                                              level, span))
        b = db.from_sequence(super_tiles)
        b.map(render_super_tile, span, output_path, shader_func, post_render_func).compute()
        results[level] = dict(success=True, stats=span, supertile_count=len(super_tiles))

    return results


def gen_super_tiles(extent, zoom_level, span=None):
    xmin, ymin, xmax, ymax = extent
    super_tile_size = min(2 ** 4 * 256,
                          (2 ** zoom_level) * 256)
    super_tile_def = MercatorTileDefinition(x_range=(xmin, xmax), y_range=(ymin, ymax),
                                            tile_size=super_tile_size)
    super_tiles = super_tile_def.get_tiles_by_extent(extent, zoom_level)
    for s in super_tiles:
        st_extent = s[3]
        x_range = (st_extent[0], st_extent[2])
        y_range = (st_extent[1], st_extent[3])
        yield {'level': zoom_level,
               'x_range': x_range,
               'y_range': y_range,
               'tile_size': super_tile_def.tile_size,
               'span': span}


def render_super_tile(tile_info, span, output_path, shader_func, post_render_func):
    level = tile_info['level']
    ds_img = shader_func(tile_info['agg'], span=span)
    return create_sub_tiles(ds_img, level, tile_info, output_path, post_render_func)


def create_sub_tiles(data_array, level, tile_info, output_path, post_render_func=None):
    # validate / createoutput_dir
    _create_dir(output_path)

    # create tile source
    tile_def = MercatorTileDefinition(x_range=tile_info['x_range'],
                                      y_range=tile_info['y_range'],
                                      tile_size=256)

    # create Tile Renderer
    if output_path.startswith('s3:'):
        renderer = S3TileRenderer(tile_def, output_location=output_path,
                                  post_render_func=post_render_func)
    else:
        renderer = FileSystemTileRenderer(tile_def, output_location=output_path,
                                          post_render_func=post_render_func)

    return renderer.render(data_array, level=level)


def invert_y_tile(y, z):
    # Convert from TMS to Google tile y coordinate, and vice versa
    return (2 ** z) - 1 - y


# TODO: change name from source to definition
class MercatorTileDefinition:
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
        self._resolutions = [
            self._get_resolution(z) for z in range(self.min_zoom, self.max_zoom + 1)]

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

        # TODO: vectorize?
        tiles = []
        for ty in range(tymin, tymax + 1):
            for tx in range(txmin, txmax + 1):
                if self.is_valid_tile(tx, ty, level):
                    t = (tx, ty, level, self.get_tile_meters(tx, ty, level))
                    tiles.append(t)

        return tiles

    def get_tile_meters(self, tx, ty, level):
        ty = invert_y_tile(ty, level)  # convert to TMS for conversion to meters
        xmin, ymin = self.pixels_to_meters(tx * self.tile_size, ty * self.tile_size, level)
        xmax, ymax = self.pixels_to_meters((tx + 1) * self.tile_size,
                                           (ty + 1) * self.tile_size, level)
        return (xmin, ymin, xmax, ymax)


class TileRenderer:

    def __init__(self, tile_definition, output_location, tile_format='PNG',
                 post_render_func=None):

        self.tile_def = tile_definition
        self.output_location = output_location
        self.tile_format = tile_format
        self.post_render_func = post_render_func

        if find_spec("PIL") is None:
            raise ImportError('pillow is required to render tiles')
        # TODO: add all the formats supported by PIL
        if self.tile_format not in ('PNG', 'JPG'):
            raise ValueError('Invalid output format')

    def render(self, da, level):
        from PIL.Image import fromarray

        xmin, xmax = self.tile_def.x_range
        ymin, ymax = self.tile_def.y_range
        extent = xmin, ymin, xmax, ymax

        tiles = self.tile_def.get_tiles_by_extent(extent, level)
        for t in tiles:
            x, y, z, data_extent = t
            dxmin, dymin, dxmax, dymax = data_extent
            arr = da.loc[{'x': slice(dxmin, dxmax), 'y': slice(dymin, dymax)}]

            if 0 in arr.shape:
                continue

            # flip since y tiles go down (Google map tiles
            img = fromarray(np.flip(arr.data, 0), 'RGBA')

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
        raise ImportError('install bokeh to enable creation of simple tile viewer')

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
        for img, x, y, z in super().render(da, level):
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
            raise ImportError('install boto3 to enable rendering to S3')

        from urllib.parse import urlparse

        s3_info = urlparse(self.output_location)
        bucket = s3_info.netloc
        client = boto3.client('s3')
        for img, x, y, z in super().render(da, level):
            tile_file_name = '{}.{}'.format(y, self.tile_format.lower())
            key = os.path.join(s3_info.path, str(z), str(x), tile_file_name).lstrip('/')
            output_buf = BytesIO()
            img.save(output_buf, self.tile_format)
            output_buf.seek(0)
            client.put_object(Body=output_buf, Bucket=bucket, Key=key, ACL='public-read')

        return 'https://{}.s3.amazonaws.com/{}'.format(bucket, s3_info.path)
