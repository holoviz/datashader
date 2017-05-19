from __future__ import absolute_import, print_function, division

import argparse
from os import path
import yaml
import uuid

from collections import OrderedDict

import pandas as pd

from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler

from bokeh.plotting import Figure
from bokeh.models import (Range1d, ImageSource, WMTSTileSource, TileRenderer, DynamicImageRenderer, Row, Column)

from bokeh.models import (Select, Slider, CheckboxGroup)

import datashader as ds
import datashader.transfer_functions as tf

from colorcet import palette

from datashader.bokeh_ext import HoverLayer, create_categorical_legend, create_ramp_legend
from datashader.utils import hold

from tornado.ioloop import IOLoop
from tornado.web import RequestHandler

from webargs import fields
from webargs.tornadoparser import use_args

# http request arguments for datashing HTTP request
ds_args = {
    'width': fields.Int(missing=800),
    'height': fields.Int(missing=600),
    'select': fields.Str(missing=""),
}


def odict_to_front(odict,key):
    """Given an OrderedDict, move the item with the given key to the front."""
    front_item = [(key,odict[key])]
    other_items = [(k,v) for k,v in odict.items() if k is not key]
    return OrderedDict(front_item+other_items)


class GetDataset(RequestHandler):
    """Handles http requests for datashading."""

    @use_args(ds_args)
    def get(self, args):

        # parse args
        selection = args['select'].strip(',').split(',')
        xmin, ymin, xmax, ymax = map(float, selection)
        self.model.map_extent = [xmin, ymin, xmax, ymax]

        glyph = self.model.glyph.get(str(self.model.field), 'points')

        # create image
        self.model.agg = self.model.create_aggregate(args['width'],
                                                     args['height'],
                                                     (xmin, xmax),
                                                     (ymin, ymax),
                                                     self.model.field,
                                                     self.model.active_axes['xaxis'],
                                                     self.model.active_axes['yaxis'],
                                                     self.model.agg_function_name, glyph)
        pix = self.model.render_image()

        def update_plots():
            self.model.update_hover()
            self.model.update_legend()

        server.get_sessions('/')[0].with_document_locked(update_plots)
        # serialize to image
        img_io = pix.to_bytesio()
        self.write(img_io.getvalue())
        self.set_header("Content-type", "image/png")


class AppState(object):
    """Simple value object to hold app state"""

    def __init__(self, config_file, outofcore, app_port):

        self.load_config_file(config_file)
        self.plot_height = 600
        self.plot_width = 990

        self.aggregate_functions = OrderedDict()
        self.aggregate_functions['Count'] = ds.count
        self.aggregate_functions['Mean'] = ds.mean
        self.aggregate_functions['Sum'] = ds.sum
        self.aggregate_functions['Min'] = ds.min
        self.aggregate_functions['Max'] = ds.max
        self.agg_function_name = list(self.aggregate_functions.keys())[0]

        # transfer function configuration
        self.transfer_functions = OrderedDict()
        self.transfer_functions['Histogram Equalization'] = 'eq_hist'
        self.transfer_functions['Linear'] = 'linear'
        self.transfer_functions['Log'] = 'log'
        self.transfer_functions[u"\u221B - Cube Root"] = 'cbrt'
        self.transfer_function = list(self.transfer_functions.values())[0]

        self.basemaps = OrderedDict()
        self.basemaps['Imagery'] = ('http://server.arcgisonline.com/arcgis'
                                    '/rest/services/World_Imagery/MapServer'
                                    '/tile/{Z}/{Y}/{X}.png')
        self.basemaps['Shaded Relief'] = ('http://services.arcgisonline.com'
                                          '/arcgis/rest/services'
                                          '/World_Shaded_Relief/MapServer'
                                          '/tile/{Z}/{Y}/{X}.png')
        self.basemaps['Toner'] = ('http://tile.stamen.com/toner-background'
                                  '/{Z}/{X}/{Y}.png')

        self.labels_url = ('http://tile.stamen.com/toner-labels'
                           '/{Z}/{X}/{Y}.png')

        self.basemap = list(self.basemaps.values())[0]

        # dynamic image configuration
        self.service_url = 'http://{host}:{port}/datashader?'
        self.service_url += 'height={HEIGHT}&'
        self.service_url += 'width={WIDTH}&'
        self.service_url += 'select={XMIN},{YMIN},{XMAX},{YMAX}&'
        self.service_url += 'cachebust={cachebust}'

        self.shader_url_vars = {}
        self.shader_url_vars['host'] = 'localhost'
        self.shader_url_vars['port'] = app_port
        self.shader_url_vars['cachebust'] = str(uuid.uuid4())

        # set defaults
        self.load_datasets(outofcore)

        # spreading
        self.spread_size = 0

        # color ramps
        default_palette = "fire"
        named_palettes = {k:p for k,p in palette.items() if not '_' in k}
        sorted_palettes = OrderedDict(sorted(named_palettes.items()))
        self.color_ramps = odict_to_front(sorted_palettes,default_palette)
        self.color_ramp  = palette[default_palette]

        self.hover_layer = None
        self.agg = None

    def load_config_file(self, config_path):
        '''load and parse yaml config file'''

        if not path.exists(config_path):
            raise IOError('Unable to find config file "{}"'.format(config_path))

        self.config_path = path.abspath(config_path)

        with open(config_path) as f:
            self.config = yaml.load(f.read())

        # parse plots
        self.axes = OrderedDict()
        for p in self.config['axes']:
            self.axes[p['name']] = p
        self.active_axes = list(self.axes.values())[0]

        # parse initial extent
        extent = self.active_axes['initial_extent']
        self.map_extent = [extent['xmin'], extent['ymin'],
                           extent['xmax'], extent['ymax']]

        # parse summary field
        self.fields = OrderedDict()
        self.colormaps = OrderedDict()
        self.color_name_maps = OrderedDict()
        self.glyph = OrderedDict()
        self.ordinal_fields = []
        self.categorical_fields = []
        for f in self.config['summary_fields']:
            self.fields[f['name']] = None if f['field'] == 'None' else f['field']

            if 'cat_colors' in f.keys():
                self.colormaps[f['name']] = f['cat_colors']
                self.categorical_fields.append(f['field'])
                self.color_name_maps[f['name']] = f['cat_names']

            elif f['field'] != 'None':
                self.ordinal_fields.append(f['field'])

            if 'glyph' in f.keys():
                self.glyph[f['field']] = f['glyph']

        self.field = list(self.fields.values())[0]
        self.field_title = list(self.fields.keys())[0]

        self.colormap = None
        if self.colormaps:
            colormap = self.colormaps.get(self.field_title, None)
            if colormap:
                self.colormap = colormap
                self.colornames = self.color_name_maps[self.field_title]

    def load_datasets(self, outofcore):
        data_path = self.config['file']
        objpath = self.config.get('objpath', None)
        print('Loading Data from {}...'.format(data_path))

        if not path.isabs(data_path):
            config_dir = path.split(self.config_path)[0]
            data_path = path.join(config_dir, data_path)

        if not path.exists(data_path):
            raise IOError('Unable to find input dataset: "{}"'.format(data_path))

        axes_fields = []
        for f in self.axes.values():
            axes_fields += [f['xaxis'], f['yaxis']]

        load_fields = [f for f in self.fields.values() if f is not None] + axes_fields

        if data_path.endswith(".csv"):
            self.df = pd.read_csv(data_path, usecols=load_fields)

            # parse categorical fields
            for f in self.categorical_fields:
                self.df[f] = self.df[f].astype('category')

        elif data_path.endswith(".h5"):
            if not objpath:
                from os.path import basename, splitext
                objpath = splitext(basename(data_path))[0]
            self.df = pd.read_hdf(data_path, objpath)

            # parse categorical fields
            for f in self.categorical_fields:
                self.df[f] = self.df[f].astype('category')

        elif data_path.endswith(".parq"):
            import dask.dataframe as dd
            self.df = dd.io.parquet.read_parquet(data_path)
            if not outofcore:
                self.df = self.df.persist()

        elif data_path.endswith(".castra"):
            import dask.dataframe as dd
            self.df = dd.from_castra(data_path)
            if not outofcore:
                self.df = self.df.cache(cache=dict)

        else:
            raise IOError("Unknown data file type; .csv and .castra currently supported")

    @hold
    def create_aggregate(self, plot_width, plot_height, x_range, y_range,
                         agg_field, x_field, y_field, agg_func, glyph):

        canvas = ds.Canvas(plot_width=plot_width,
                           plot_height=plot_height,
                           x_range=x_range,
                           y_range=y_range)

        method = getattr(canvas, glyph)

        # handle categorical field
        if agg_field in self.categorical_fields:
            agg = method(self.df, x_field, y_field, ds.count_cat(agg_field))

        # handle ordinal field
        elif agg_field in self.ordinal_fields:
            func = self.aggregate_functions[agg_func]
            agg = method(self.df, x_field, y_field, func(agg_field))
        else:
            agg = method(self.df, x_field, y_field)

        return agg

    def render_image(self):
        if self.colormaps:
            colormap = self.colormaps.get(self.field_title, None)
            if colormap:
                self.colormap = colormap
                self.colornames = self.color_name_maps[self.field_title]

        pix = tf.shade(self.agg, cmap=self.color_ramp, color_key=self.colormap, how=self.transfer_function)

        if self.spread_size > 0:
            pix = tf.spread(pix, px=self.spread_size)

        return pix

    def update_hover(self):

        # hover (temporarily disabled)
        return

        if not self.hover_layer:
            self.hover_layer = HoverLayer(field_name=self.field_title,
                                          extent=self.map_extent,
                                          is_categorical=self.field in self.categorical_fields,
                                          agg=self.agg)
            self.fig.renderers.append(self.hover_layer.renderer)
            self.fig.add_tools(self.hover_layer.tool)
        else:
            self.hover_layer.is_categorical = self.field in self.categorical_fields
            self.hover_layer.extent = self.map_extent
            self.hover_layer.agg = self.agg

    def update_legend(self):

        # legends (temporarily disabled)
        return

        if self.field in self.categorical_fields:
            cat_legend = create_categorical_legend(self.colormap, aliases=self.colornames)
            self.legend_side_vbox.children = [cat_legend]
            self.legend_bottom_vbox.children = []

        else:
            legend_fig = create_ramp_legend(self.agg,
                                            self.color_ramp,
                                            how=self.transfer_function,
                                            width=self.plot_width)

            self.legend_bottom_vbox = [legend_fig]
            self.legend_side_vbox.children = []


class AppView(object):
    def __init__(self, app_model):
        self.model = app_model
        self.create_layout()

    def create_layout(self):
        # create figure
        self.x_range = Range1d(start=self.model.map_extent[0],
                               end=self.model.map_extent[2], bounds=None)
        self.y_range = Range1d(start=self.model.map_extent[1],
                               end=self.model.map_extent[3], bounds=None)

        self.fig = Figure(tools='wheel_zoom,pan',
                          x_range=self.x_range,
                          lod_threshold=None,
                          plot_width=self.model.plot_width,
                          plot_height=self.model.plot_height,
                          background_fill_color='black',
                          y_range=self.y_range)

        self.fig.min_border_top = 0
        self.fig.min_border_bottom = 10
        self.fig.min_border_left = 0
        self.fig.min_border_right = 0
        self.fig.axis.visible = False

        self.fig.xgrid.grid_line_color = None
        self.fig.ygrid.grid_line_color = None

        # add tiled basemap
        self.tile_source = WMTSTileSource(url=self.model.basemap)
        self.tile_renderer = TileRenderer(tile_source=self.tile_source)
        self.fig.renderers.append(self.tile_renderer)

        # add datashader layer
        self.image_source = ImageSource(url=self.model.service_url,
                                        extra_url_vars=self.model.shader_url_vars)
        self.image_renderer = DynamicImageRenderer(image_source=self.image_source)
        self.fig.renderers.append(self.image_renderer)

        # add label layer
        self.label_source = WMTSTileSource(url=self.model.labels_url)
        self.label_renderer = TileRenderer(tile_source=self.label_source)
        self.fig.renderers.append(self.label_renderer)

        # Add placeholder for legends (temporarily disabled)
        # self.model.legend_side_vbox = Column()
        # self.model.legend_bottom_vbox = Column()

        # add ui components
        controls = []
        axes_select = Select(name='Axes', options=list(self.model.axes.keys()))
        axes_select.on_change('value', self.on_axes_change)
        controls.append(axes_select)

        self.field_select = Select(name='Field', options=list(self.model.fields.keys()))
        self.field_select.on_change('value', self.on_field_change)
        controls.append(self.field_select)

        self.aggregate_select = Select(name='Aggregate', options=list(self.model.aggregate_functions.keys()))
        self.aggregate_select.on_change('value', self.on_aggregate_change)
        controls.append(self.aggregate_select)

        transfer_select = Select(name='Transfer Function',
                                 options=list(self.model.transfer_functions.keys()))
        transfer_select.on_change('value', self.on_transfer_function_change)
        controls.append(transfer_select)

        color_ramp_select = Select(name='Color Ramp', options=list(self.model.color_ramps.keys()))
        color_ramp_select.on_change('value', self.on_color_ramp_change)
        controls.append(color_ramp_select)

        spread_size_slider = Slider(title="Spread Size (px)", value=0, start=0,
                                    end=10, step=1)
        spread_size_slider.on_change('value', self.on_spread_size_change)
        controls.append(spread_size_slider)

        # hover (temporarily disabled)
        #hover_size_slider = Slider(title="Hover Size (px)", value=8, start=4,
        #                           end=30, step=1)
        #hover_size_slider.on_change('value', self.on_hover_size_change)
        #controls.append(hover_size_slider)

        # legends (temporarily disabled)
        # controls.append(self.model.legend_side_vbox)

        # add map components
        basemap_select = Select(name='Basemap', value='Imagery', options=list(self.model.basemaps.keys()))
        basemap_select.on_change('value', self.on_basemap_change)

        image_opacity_slider = Slider(title="Opacity", value=100, start=0,
                                      end=100, step=1)
        image_opacity_slider.on_change('value', self.on_image_opacity_slider_change)

        basemap_opacity_slider = Slider(title="Basemap Opacity", value=100, start=0,
                                        end=100, step=1)
        basemap_opacity_slider.on_change('value', self.on_basemap_opacity_slider_change)

        show_labels_chk = CheckboxGroup(labels=["Show Labels"], active=[0])
        show_labels_chk.on_click(self.on_labels_change)

        map_controls = [basemap_select, basemap_opacity_slider,
                        image_opacity_slider, show_labels_chk]

        self.controls = Column(height=600, children=controls)
        self.map_controls = Row(width=self.fig.plot_width, children=map_controls)

        # legends (temporarily disabled)
        self.map_area = Column(width=900, height=600,
                               children=[self.map_controls, self.fig])
        self.layout = Row(width=1300, height=600,
                          children=[self.controls, self.map_area])
        self.model.fig = self.fig
        self.model.update_hover()

    def update_image(self):
        self.model.shader_url_vars['cachebust'] = str(uuid.uuid4())
        self.image_renderer.image_source = ImageSource(url=self.model.service_url,
                                                       extra_url_vars=self.model.shader_url_vars)

    def on_field_change(self, attr, old, new):
        self.model.field_title = new
        self.model.field = self.model.fields[new]

        self.model.hover_layer.field_name = new
        self.model.hover_layer.is_categorical = self.model.field in self.model.categorical_fields
        self.update_image()

        if not self.model.field:
            self.aggregate_select.options = [("No Aggregates Available", "")]
        elif self.model.field in self.model.categorical_fields:
            self.model.hover_layer.is_categorical = True
            self.aggregate_select.options = [("Categorical", "count_cat")]
        else:
            opts = [(k, k) for k in self.model.aggregate_functions.keys()]
            self.aggregate_select.options = opts
            self.model.hover_layer.is_categorical = False

    def on_basemap_change(self, attr, old, new):
        self.model.basemap = self.model.basemaps[new]
        self.tile_renderer.tile_source = WMTSTileSource(url=self.model.basemap)

    def on_hover_size_change(self, attr, old, new):
        self.model.hover_layer.size = int(new)

    def on_spread_size_change(self, attr, old, new):
        self.model.spread_size = int(new)
        self.update_image()

    def on_axes_change(self, attr, old, new):
        self.model.active_axes = self.model.axes[new]
        self.update_image()

    def on_aggregate_change(self, attr, old, new):
        self.model.agg_function_name = new
        self.update_image()

    def on_transfer_function_change(self, attr, old, new):
        self.model.transfer_function = self.model.transfer_functions[new]
        self.update_image()

    def on_color_ramp_change(self, attr, old, new):
        self.model.color_ramp = self.model.color_ramps[new]
        self.update_image()

    def on_image_opacity_slider_change(self, attr, old, new):
        self.image_renderer.alpha = new / 100

    def on_basemap_opacity_slider_change(self, attr, old, new):
        self.tile_renderer.alpha = new / 100

    def on_labels_change(self, new):
        self.label_renderer.alpha = 1 if new else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='yaml config file (e.g. nyc_taxi.yml)', required=True)
    parser.add_argument('-p', '--port', help='port number to use for communicating with server; defaults to 5000', default=5000)
    parser.add_argument('-o', '--outofcore', help='use out-of-core processing if available, for datasets larger than memory',
                        default=False, action='store_true')
    args = vars(parser.parse_args())

    APP_PORT = args['port']

    def add_roots(doc):
        model = AppState(args['config'], args['outofcore'], APP_PORT)
        view = AppView(model)
        GetDataset.model = model
        doc.add_root(view.layout)

    app = Application(FunctionHandler(add_roots))

    # Start server object wired to bokeh client. Instantiating ``Server``
    # directly is used to add custom http endpoint into ``extra_patterns``.
    url = 'http://localhost:{}/'.format(APP_PORT)
    print('Starting server at {}...'.format(url))

    io_loop = IOLoop.current()

    server = Server({'/': app}, io_loop=io_loop, extra_patterns=[('/datashader', GetDataset)], port=APP_PORT)
    server.start()

    io_loop.add_callback(server.show, "/")
    io_loop.start()
