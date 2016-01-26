from __future__ import absolute_import, print_function, division

from os import path
import yaml
import pdb

from collections import OrderedDict

import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd

from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler

from bokeh.plotting import Figure
from bokeh.models import (Range1d, ImageSource, WMTSTileSource, TileRenderer,
                          DynamicImageRenderer)
from bokeh.models.widgets.layouts import HBox, VBox
from bokeh.models.widgets import Select, Slider

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

class GetDataset(RequestHandler):
    """Handles http requests for datashading."""
    @use_args(ds_args)
    def get(self, args):
        # parse args
        selection = args['select'].strip(',').split(',')
        xmin, ymin, xmax, ymax = map(float, selection)
        self.model.map_extent = [xmin, ymin, xmax, ymax]

        # create image
        cvs = ds.Canvas(plot_width=args['width'],
                        plot_height=args['height'],
                        x_range=(xmin, xmax),
                        y_range=(ymin, ymax))
        agg = cvs.points(self.model.df,
                         self.model.location[1],
                         self.model.location[2],
                         agg=self.model.aggregate_function(self.model.field))
        pix = tf.interpolate(agg.agg, (255, 204, 204), 'red',
                             how=self.model.transfer_function)

        # serialize to image
        img_io = pix.to_bytesio()
        self.write(img_io.getvalue())
        self.set_header("Content-type", "image/png")


class AppState(object):
    """Simple value object to hold app state"""

    def __init__(self, config_file='nyc_taxi.yml'):

        self.load_config_file(config_file)

        self.aggregate_functions = OrderedDict()
        self.aggregate_functions['Count'] = ds.count
        self.aggregate_functions['Mean'] = ds.mean
        self.aggregate_functions['Sum'] = ds.sum
        self.aggregate_functions['Min'] = ds.min
        self.aggregate_functions['Max'] = ds.max
        self.aggregate_function = self.aggregate_functions.values()[0]

        # transfer function configuration
        self.transfer_functions = OrderedDict()
        self.transfer_functions['Log'] = 'log'
        self.transfer_functions['Linear'] = 'linear'
        self.transfer_functions[u"\u221B"] = 'cbrt'
        self.transfer_function = self.transfer_functions.values()[0]

        # map configurations
        self.map_extent = [-8240227.037, 4974203.152, -8231283.905, 4979238.441]

        self.basemaps = OrderedDict()
        self.basemaps['Toner'] = ('http://tile.stamen.com/toner-background'
                                  '/{Z}/{X}/{Y}.png')
        self.basemaps['Imagery'] = ('http://server.arcgisonline.com/ArcGIS'
                                    '/rest/services/World_Imagery/MapServer'
                                    '/tile/{Z}/{Y}/{X}.png')
        self.basemaps['Shaded Relief'] = ('http://services.arcgisonline.com'
                                          '/ArcGIS/rest/services'
                                          '/World_Shaded_Relief/MapServer'
                                          '/tile/{Z}/{Y}/{X}.png')
        self.basemap = self.basemaps.values()[0]

        # dynamic image configuration
        self.service_url = 'http://{host}:{port}/datashader?'
        self.service_url += 'height={HEIGHT}&'
        self.service_url += 'width={WIDTH}&'
        self.service_url += 'select={XMIN},{YMIN},{XMAX},{YMAX}'

        self.shader_url_vars = {}
        self.shader_url_vars['host'] = 'localhost'
        self.shader_url_vars['port'] = 5000

        # set defaults
        self.load_datasets()

    def load_config_file(self, config_path):
        '''load and parse yaml config file'''

        if not path.exists(config_path):
            raise IOError('Unable to find config file "{}"'.format(config_path))

        with open(config_path) as f:
            self.config = yaml.load(f.read())

        self.locations = OrderedDict()
        for p in self.config['plots']:
            self.locations[p['name']] = (p['name'], p['xaxis'], p['yaxis'])
        self.location = self.locations.values()[0]

        self.fields = OrderedDict()
        for f in self.config['summary_fields']:
            self.fields[f['name']] = f['field']
        self.field = self.fields.values()[0]

    def load_datasets(self):
        print('Loading Data...')
        examples_dir = path.dirname(path.realpath(__file__))
        taxi_path = path.join(examples_dir, 'data', 'taxi.csv')
        if path.exists(taxi_path):
            location_fields = []
            for f in self.locations.values():
                location_fields += [f[1], f[2]]

            load_fields = self.fields.values() + location_fields
            self.df = pd.read_csv(taxi_path, usecols=load_fields)
        else:
            raise IOError('Unable to find input dataset')

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

        self.fig = Figure(tools='wheel_zoom,pan', x_range=self.x_range,
                          y_range=self.y_range)
        self.fig.plot_height = 660
        self.fig.plot_width = 990
        self.fig.axis.visible = False

        # add tiled basemap
        self.tile_source = WMTSTileSource(url=self.model.basemap)
        self.tile_renderer = TileRenderer(tile_source=self.tile_source)
        self.fig.renderers.append(self.tile_renderer)

        # add datashader layer
        self.image_source = ImageSource(url=self.model.service_url,
                                        extra_url_vars=self.model.shader_url_vars)
        self.image_renderer = DynamicImageRenderer(image_source=self.image_source)
        self.fig.renderers.append(self.image_renderer)

        # add ui components
        location_select = Select.create(name='Location',
                                        options=self.model.locations)
        location_select.on_change('value', self.on_location_change)

        field_select = Select.create(name='Field', options=self.model.fields)
        field_select.on_change('value', self.on_field_change)

        aggregate_select = Select.create(name='Aggregate',
                                         options=self.model.aggregate_functions)
        aggregate_select.on_change('value', self.on_aggregate_change)

        transfer_select = Select.create(name='Transfer Function',
                                        options=self.model.transfer_functions)
        transfer_select.on_change('value', self.on_transfer_function_change)

        basemap_select = Select.create(name='Basemap', value='Toner',
                                       options=self.model.basemaps)
        basemap_select.on_change('value', self.on_basemap_change)

        opacity_slider = Slider(title="Opacity", value=100, start=0,
                                end=100, step=1)
        opacity_slider.on_change('value', self.on_opacity_slider_change)

        controls = [location_select, field_select, aggregate_select,
                    transfer_select, basemap_select, opacity_slider]
        self.controls = HBox(width=self.fig.plot_width, children=controls)
        self.layout = VBox(width=self.fig.plot_width,
                           height=self.fig.plot_height,
                           children=[self.controls, self.fig])

    def update_image(self):
        for renderer in self.fig.renderers:
            if hasattr(renderer, 'image_source'):
                renderer.image_source = ImageSource(url=self.model.service_url,
                        extra_url_vars=self.model.shader_url_vars)
                break

    def update_tiles(self):
        for renderer in self.fig.renderers:
            if hasattr(renderer, 'tile_source'):
                renderer.tile_source = WMTSTileSource(url=self.model.basemap)

    def on_field_change(self, attr, old, new):
        self.model.field = self.model.fields[new]
        self.update_image()

    def on_basemap_change(self, attr, old, new):
        self.model.basemap = self.model.basemaps[new]
        self.update_tiles()

    def on_location_change(self, attr, old, new):
        self.model.location = self.model.locations[new]
        self.update_image()

    def on_aggregate_change(self, attr, old, new):
        self.model.aggregate_function = self.model.aggregate_functions[new]
        self.update_image()

    def on_transfer_function_change(self, attr, old, new):
        self.model.transfer_function = self.model.transfer_functions[new]
        self.update_image()

    def on_opacity_slider_change(self, attr, old, new):
        for renderer in self.fig.renderers:
            if hasattr(renderer, 'image_source'):
                renderer.alpha = new / 100


if __name__ == '__main__':
    def add_roots(doc):
        model = AppState()
        view = AppView(model)
        GetDataset.model = model
        doc.add_root(view.layout)

    app = Application()
    app.add(FunctionHandler(add_roots))
    # Start server object wired to bokeh client. Instantiating ``Server``
    # directly is used to add custom http endpoint into ``extra_patterns``.
    server = Server(app, io_loop=IOLoop(),
                    extra_patterns=[(r"/datashader", GetDataset)], port=5000)
    print('Starting server at http://localhost:5000/...')
    server.start()
