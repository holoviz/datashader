from __future__ import absolute_import, print_function, division

import os

import datashader as ds
import datashader.transfer_functions as tf
import dask.dataframe as dd

from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.plotting import Figure
from bokeh.models import Range1d, ImageSource, WMTSTileSource, TileRenderer, DynamicImageRenderer
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
    '''handles http requests for datashading using input arguments listed above in ``ds_args``
    '''

    @use_args(ds_args)
    def get(self, args):

        # parse args -----------------------------
        xmin, ymin, xmax, ymax = map(float, args['select'].strip(',').split(','))
        self.model.map_extent = [xmin, ymin, xmax, ymax]

        # create aggregate -----------------------
        cvs = ds.Canvas(plot_width=args['width'], plot_height=args['height'], x_range=(xmin, xmax), y_range=(ymin, ymax), stretch=self.model.allow_stretch)
        if self.model.transfer_function == 'count':
            agg = cvs.points(self.model.df, self.model.dataset[1], self.model.dataset[2], count=ds.count('passenger_count'))
            pix = tf.interpolate(agg.count, (255, 204, 204), 'red')

        # TODO: work out transfer functions
        elif self.model.transfer_function == 'mean':
            agg = cvs.points(self.model.df, self.model.dataset[1], self.model.dataset[2], mean=ds.mean('passenger_count'))
            pix = tf.interpolate(agg.mean, (255, 204, 204), 'red')

        # serialize to image --------------------
        img_io = pix.to_bytesio()
        self.write(img_io.getvalue())
        self.set_header("Content-type", "image/png")

class AppState(object):
    '''simple value object to hold app state
    '''
    def __init__(self):

        # data configurations --------------------
        self.datasets = {}
        self.datasets['NYC Taxi Pickups'] = ('TAXI_PICKUP', 'pickup_longitude', 'pickup_latitude')
        self.datasets['NYC Taxi Dropoffs'] = ('TAXI_DROPOFF', 'dropoff_longitude', 'dropoff_latitude')
        self.dataset = self.datasets['NYC Taxi Pickups']

        # transfer function configuration -------------
        self.transfer_functions = {}
        self.transfer_functions['Count'] = 'count'
        self.transfer_functions['Mean'] = 'mean'
        self.transfer_function = self.transfer_functions['Count']

        # map configurations --------------------
        self.map_extent = [-8242056, 4971273, -8227397, 4982613]

        self.basemaps = {}
        self.basemaps['Toner'] = 'http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png'
        self.basemaps['Imagery'] = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}'
        self.basemaps['Shaded Relief'] = 'http://services.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{Z}/{Y}/{X}'
        self.basemap = self.basemaps['Toner']

        # dynamic image configuration --------------------
        self.service_url = 'http://{host}:{port}/datashader?'
        self.service_url += 'height={HEIGHT}&'
        self.service_url += 'width={WIDTH}&'
        self.service_url += 'arl={arl}&'
        self.service_url += 'select={XMIN},{YMIN},{XMAX},{YMAX}'

        self.shader_url_vars = {}
        self.shader_url_vars['host'] = 'localhost'
        self.shader_url_vars['port'] = 5000
        self.shader_url_vars['layer_name'] = self.dataset
        self.shader_url_vars['arl'] = 'stub'

        self.allow_stretch_options = {}
        self.allow_stretch_options['Yes'] = True
        self.allow_stretch_options['No'] = False
        self.allow_stretch = False

        # set defaults ------------------------
        self.cache = {}
        self.load_datasets()

    def load_datasets(self):
        print('Loading Data...')
        if os.path.exists('data/taxi.castra'):
            df = dd.from_castra('data/taxi.castra')
            df = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
            df = df.compute()
            self.cache['TAXI_PICKUP'] = df
            self.cache['TAXI_DROPOFF'] = df
        else:
            raise FileNotFoundError('Unable to find input dataset')

    @property
    def df(self):
        return self.cache[self.dataset[0]]

class AppView(object):

    def __init__(self, app_model):
        self.model = app_model
        self.create_layout()
        self.update()

    def create_layout(self):

        # create figure ----------------------------
        self.x_range = Range1d(start=self.model.map_extent[0], end=self.model.map_extent[2], bounds=None)
        self.y_range = Range1d(start=self.model.map_extent[1], end=self.model.map_extent[3], bounds=None)

        self.fig = Figure(tools='wheel_zoom,pan', x_range=self.x_range, y_range=self.y_range)
        self.fig.plot_height = 660
        self.fig.plot_width = 990
        self.fig.axis.visible = False

        # add tiled basemap ----------------------------
        self.tile_source = WMTSTileSource(url=self.model.basemap)
        self.tile_renderer = TileRenderer(tile_source=self.tile_source)
        self.fig.renderers.append(self.tile_renderer)

        # add datashader layer ----------------------------
        self.image_source = ImageSource(**dict(url=self.model.service_url, extra_url_vars=self.model.shader_url_vars))
        self.image_renderer = DynamicImageRenderer(image_source=self.image_source)
        self.fig.renderers.append(self.image_renderer)

        # add ui components ----------------------------
        dataset_select = Select.create(name='Dataset', options=self.model.datasets)
        dataset_select.on_change('value', self.on_dataset_change)

        transfer_select = Select.create(name='Transfer Function', options=self.model.transfer_functions)
        transfer_select.on_change('value', self.on_transfer_function_change)

        allow_stretch_select = Select.create(name='Allow Stretch', options=self.model.allow_stretch_options)
        allow_stretch_select.on_change('value', self.on_allow_stretch_change)

        basemap_select = Select.create(name='Basemap', value='Toner', options=self.model.basemaps)
        basemap_select.on_change('value', self.on_basemap_change)

        opacity_slider = Slider(title="Opacity", value=100, start=0, end=100, step=1)
        opacity_slider.on_change('value', self.on_opacity_slider_change)

        self.controls = HBox(width=self.fig.plot_width, children=[dataset_select, transfer_select, allow_stretch_select, basemap_select, opacity_slider])
        self.layout = VBox(width=self.fig.plot_width, height=self.fig.plot_height, children=[self.controls, self.fig])

    def update(self):
        '''configures and returns bokeh user interface for datashader map.
        '''
        pass

    def update_image(self):
        for i in range(len(self.fig.renderers)):
            if hasattr(self.fig.renderers[i], 'image_source'):
                self.fig.renderers[i].update(image_source=ImageSource(**dict(url=self.model.service_url, extra_url_vars=self.model.shader_url_vars)))
                self.fig.x_range = Range1d(start=-20000000, end=20000000, bounds=None)

    def update_tiles(self):
        for i in range(len(self.fig.renderers)):
            if hasattr(self.fig.renderers[i], 'tile_source'):
                self.fig.renderers[i].update(tile_source=WMTSTileSource(url=self.model.basemap))
                self.fig.x_range = Range1d(start=-20000000, end=20000000, bounds=None)

    def on_basemap_change(self, attr, old, new):
        self.model.basemap = self.model.basemaps[new]
        self.update_tiles()

    def on_dataset_change(self, attr, old, new):
        self.model.dataset = self.model.datasets[new]
        self.update_image()

    def on_transfer_function_change(self, attr, old, new):
        self.model.transfer_function = self.model.transfer_functions[new]
        self.update_image()

    def on_allow_stretch_change(self, attr, old, new):
        self.model.allow_stretch = self.model.allow_stretch_options[new]
        self.update_image()

    def on_opacity_slider_change(self, attr, old, new):
        for i in range(len(self.fig.renderers)):
            if hasattr(self.fig.renderers[i], 'image_source'):
                self.fig.renderers[i].alpha = new / 100
                self.fig.x_range = Range1d(start=-20000000, end=20000000, bounds=None)


# ------------ entry point ---------------
def main():
    '''starts server object wired to bokeh client. Instantiating ``Server`` directly is
       used to add custom http endpoint into ``extra_patterns``.
    '''
    def add_roots(doc):
        model = AppState()
        view = AppView(model)
        GetDataset.model = model
        doc.add_root(view.layout)

    app = Application()
    app.add(FunctionHandler(add_roots))
    server = Server(app, io_loop=IOLoop(), extra_patterns=[(r"/datashader", GetDataset)], port=5000)
    print('Starting server at http://localhost:5000/...')
    server.start()

if __name__ == '__main__':
    main()
