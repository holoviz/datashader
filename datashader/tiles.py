from __future__ import absolute_import, division, print_function

import math
import os

import dask
import dask.bag as db

from PIL.Image import fromarray


__all__ = ['render_tiles', 'MercatorTileDefinition', 'MercatorSuperTileDefinition']


# helpers ---------------------------------------------------------------------
def _create_dir(path):

    import os, errno

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _get_super_tile_min_max(tile_info, load_data_func, ds_pipeline_func):
    tile_size = tile_info['tile_size']
    df = load_data_func(tile_info['x_range'], tile_info['y_range'])
    agg, _ = ds_pipeline_func(df, x_range=tile_info['x_range'],
                              y_range=tile_info['y_range'],
                              plot_height=tile_size, plot_width=tile_size)
    return [agg.data.min(), agg.data.max()]


def calculate_zoom_level_stats(full_extent, level, load_data_func, ds_pipeline_func,
                               color_ranging_strategy='fullscan'):
    if color_ranging_strategy == 'fullscan':
        b = db.from_sequence(list(gen_super_tiles(full_extent, level)))
        b = b.map(_get_super_tile_min_max, load_data_func, ds_pipeline_func).flatten().distinct()
        return dask.compute(b.min(), b.max())
    else:
        raise ValueError('Invalid color_ranging_strategy option')


def render_tiles(full_extent, levels, load_data_func, ds_pipeline_func,
                 post_render_func, output_path, color_ranging_strategy='fullscan'):

    #TODO: get full extent once at beginning for all levels

    print(levels)
    for level in levels:
        print(level)

#        span = calculate_zoom_level_stats(full_extent, level,
#                                          load_data_func, ds_pipeline_func,
#                                          color_ranging_strategy='fullscan')

        results = []
        super_tiles = list(gen_super_tiles(full_extent, level))
        for st in super_tiles:
            print(st)
            r = render_super_tile(st, output_path, load_data_func, ds_pipeline_func, post_render_func)
            results.append(r)

    return results

    '''

    b = db.from_sequence()
    return b.map(render_super_tile, output_path, load_data_func, ds_pipeline_func, post_render_func).compute()

    '''

def gen_super_tiles(extent, zoom_level):

    # TODO: decide whether to use x_range/y_range style or xmin, ymin, xmax, ymax extent lists
    xmin, ymin, xmax, ymax = extent
    super_tile_def = MercatorSuperTileDefinition(x_range=(xmin, xmax), y_range=(ymin, ymax))
    super_tiles = super_tile_def.get_tiles_by_extent(extent, zoom_level)
    for s in super_tiles:
        st_extent = s[3]
        x_range = (st_extent[0], st_extent[2])
        y_range = (st_extent[1], st_extent[3])
        yield {'level': zoom_level, 'x_range': x_range, 'y_range': y_range, 'tile_size': super_tile_def.tile_size}


def render_super_tile(tile_info, output_path, load_data_func, ds_pipeline_func, post_render_func):
    tile_size = tile_info['tile_size']
    level = tile_info['level']
    df = load_data_func(tile_info['x_range'], tile_info['y_range'])
    agg, ds_img = ds_pipeline_func(df, x_range=tile_info['x_range'], y_range=tile_info['y_range'],
                                   plot_height=tile_size, plot_width=tile_size)
    return create_sub_tiles(ds_img, level, tile_info, output_path, post_render_func)


def create_sub_tiles(data_array, level, tile_info, output_path, post_render_func=None):

    # validate / createoutput_dir
    _create_dir(output_path)

    # create tile source
    tile_def = MercatorTileDefinition(x_range=tile_info['x_range'],
                                      y_range=tile_info['y_range'])

    # create Tile Renderer
    if output_path.startswith('s3://'):
        renderer = S3TileRenderer(tile_def)
    else:
        renderer = FileSystemTileRenderer(tile_def, output_location=output_path)

    return renderer.render(data_array, level=level)


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
        self._resolutions = [self._get_resolution(z) for z in range(self.min_zoom, self.max_zoom+1)]

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
        return (i-1)


    def get_closest_level_by_extent(self, extent, height, width):

        x_rs = (extent[2] - extent[0]) / width
        y_rs = (extent[3] - extent[1]) / height
        resolution = max(x_rs, y_rs)

        def _close_reducer(previous, current):

            if abs(current - resolution) < abs(previous - resolution):
                return current

            return previous

        closest = self._resolutions.reduce(_close_reducer, self._resolutions)
        return self._resolutions.index(closest)


    def snap_to_zoom_level(self, extent, height, width, level):
        xmin, ymin, xmax, ymax = extent
        desired_res = self._resolutions[level]
        desired_x_delta = width * desired_res
        desired_y_delta = height * desired_res

        if not self.snap_to_zoom:
            xscale = (xmax - xmin) / desired_x_delta
            yscale = (ymax - ymin) / desired_y_delta

            if (xscale > yscale):
                desired_x_delta = xmax - xmin
                desired_y_delta = desired_y_delta * xscale
            else:
                desired_x_delta = desired_x_delta * yscale
                desired_y_delta = ymax - ymin

        x_adjust = (desired_x_delta - (xmax - xmin)) / 2
        y_adjust = (desired_y_delta - (ymax - ymin)) / 2

        return (xmin - x_adjust, ymin - y_adjust, xmax + x_adjust, ymax + y_adjust)


    def tms_to_wmts(self, x, y, z):
        'Note this works both ways'
        return (x, math.pow(2, z) - 1 - y, z)


    def wmts_to_tms(self, x, y, z):
        'Note this works both ways'
        return (x, math.pow(2, z) - 1 - y, z)


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


    def pixels_to_tile(self, px, py):
        tx = math.ceil(px / self.tile_size)
        tx = tx if tx == 0 else tx - 1
        ty = max(math.ceil(py / self.tile_size) - 1, 0)
        return (tx, ty)


    def pixels_to_raster(self, px, py, level):
        map_size = self.tile_size << level
        return (px, map_size - py)


    def meters_to_tile(self, mx, my, level):
        px, py = self.meters_to_pixels(mx, my, level)
        return self.pixels_to_tile(px, py)


    def get_tiles_by_extent(self, extent, level, tile_border=1):

        # unpack extent and convert to tile coordinates
        xmin, ymin, xmax, ymax = extent
        txmin, tymin = self.meters_to_tile(xmin, ymin, level)
        txmax, tymax = self.meters_to_tile(xmax, ymax, level)

        # add tiles which border
        txmin -= tile_border
        tymin -= tile_border
        txmax += tile_border
        tymax += tile_border

        # TODO: vectorize?
        tiles = []
        for ty in range(tymin, tymax + 1):
            for tx in range(txmin, txmax + 1):
                if self.is_valid_tile(tx, ty, level):
                    t = (tx, ty, level, self.get_tile_meters(tx, ty, level))
                    tiles.append(t)

        return tiles


    def get_tile_meters(self, tx, ty, level):
        xmin, ymin = self.pixels_to_meters(tx * self.tile_size, ty * self.tile_size, level)
        xmax, ymax = self.pixels_to_meters((tx + 1) * self.tile_size, (ty + 1) * self.tile_size, level)
        return (xmin, ymin, xmax, ymax)


class MercatorSuperTileDefinition(MercatorTileDefinition):

    def __init__(self, x_range, y_range):
        super(MercatorSuperTileDefinition, self).__init__(x_range=x_range,
                                                          y_range=y_range,
                                                          tile_size=2560)


#  TILE RENDERER -----------------------------------------------------------------------------------

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

    def adjust_image(self, image, anchor='left'):

        from PIL import Image

        tile_size = self.tile_def.tile_size

        if anchor in ('left', 'top-left', 'top'):
            anchor_coords = (0, 0)

        elif anchor in ('right', 'top-right'):
            anchor_coords = (tile_size - (tile_size - image.width), 0)

        elif anchor in ('bottom-left', 'bottom'):
            anchor_coords = (0, tile_size - (tile_size - image.height))

        elif anchor in ('bottom-right'):
            anchor_coords = (tile_size - (tile_size - image.width),
                             tile_size - (tile_size - image.height))

        elif anchor in ('center'):
            left = 0
            upper = 0
            right = 0
            lower = 0
            anchor_coords = (left, upper, right, lower)
            raise NotImplementedError('center (single tile edge case) not implemented yet')

        else:
            raise ValueError('Invalid anchor argument: {}'.format(anchor))

        new_image = Image.new('RGBA', (tile_size, tile_size), (0, 0, 0, 0))
        new_image.paste(image, anchor_coords)
        return new_image


class FileSystemTileRenderer(TileRenderer):


    def render(self, da, level):
        print('.render(level={})'.format(level))

        tile_width = self.tile_def.tile_size
        tile_height =  self.tile_def.tile_size
        xmin, xmax = self.tile_def.x_range
        ymin, ymax = self.tile_def.y_range
        extent = xmin, ymin, xmax, ymax

        tiles = self.tile_def.get_tiles_by_extent(extent, level)
        path_template = '{}/{}/{}.{}'
        for t in tiles:
            x, y, z, data_extent = t
            dxmin, dymin, dxmax, dymax = data_extent

            is_top = False
            is_bottom = False
            is_right = False
            is_left = False

            if xmin in data_extent:
                is_left = True

            if ymin in data_extent:
                is_bottom = True

            if xmax in data_extent:
                is_right = True

            if ymax in data_extent:
                is_top = True


            if is_top and is_left:
                anchor = 'top-left'

            elif is_top and is_right:
                anchor = 'top-right'

            elif is_bottom and is_left:
                anchor = 'bottom-left'

            elif is_bottom and is_right:
                anchor = 'bottom-right'

            elif is_bottom:
                anchor = 'bottom'

            elif is_left:
                anchor = 'left'

            elif is_right:
                anchor = 'right'

            elif all([is_left, is_right, is_top, is_bottom]):
                anchor = 'center' # TODO: need to implement

            elif not any([is_left, is_right, is_top, is_bottom]):
                anchor = None


            tile_file_name = '{}.{}'.format(y, self.tile_format.lower())
            tile_directory = os.path.join(self.output_location, str(z), str(x))
            _create_dir(tile_directory)
            output_file = os.path.join(tile_directory, tile_file_name)
            arr = da.loc[{'x':slice(dxmin, dxmax), 'y':slice(dymin, dymax)}]

            img = fromarray(arr.data, 'RGBA')

            # START HERE AND TEST
            if anchor:
                img = self.adjust_image(img, anchor)

            if self.post_render_func:
                img = self.post_render_func(img)

            img.save(output_file, self.tile_format)


class S3TileRenderer(TileRenderer):

    def render(tile_obj):
        raise NotImplementedError('S3TileRenderer not yet implemented')
