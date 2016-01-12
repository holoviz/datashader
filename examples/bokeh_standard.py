from __future__ import absolute_import, print_function, division

import os
import logging
import datetime as dt

import datashader as ds
import datashader.transfer_functions as tf
import dask.dataframe as dd
import numpy as np
from dynd import nd

from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.tile_providers import STAMEN_TONER
from bokeh.plotting import figure, show, output_file
from bokeh.models import Range1d
from bokeh.models import ImageSource
from bokeh.models.widgets.layouts import HBox, VBox
from bokeh.models.widgets import Select

from tornado.ioloop import IOLoop
from tornado.web import RequestHandler

from fastcache import lru_cache

from webargs import fields
from webargs.tornadoparser import use_args

try:
    from urllib2 import unquote
except ImportError:
    from urllib.parse import unquote

ds_args = {
    'width': fields.Int(missing=800),
    'height': fields.Int(missing=600),
    'requesterID': fields.Str(missing=""),
    'allowStretch': fields.Bool(missing=True),
    'aggregator': fields.Str(missing="COC_COMP"),
    'arl': fields.Str(missing=""),
    'select': fields.Str(missing=""),
}

print('Loading Data...')
df = dd.from_castra('data/taxi.castra')
df = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
df = df.compute()

class GetDataset(RequestHandler):

    @use_args(ds_args)
    def get(self, dataset, args):
        xmin, ymin, xmax, ymax = map(float, args['select'].strip(',').split(','))
        cvs = ds.Canvas(plot_width=args['width'], plot_height=args['height'], x_range=(xmin, xmax), y_range=(ymin, ymax), stretch=False)
        pickups = cvs.points(df, 'pickup_longitude', 'pickup_latitude',
                             count=ds.count('passenger_count'))
        dropoffs = cvs.points(df, 'dropoff_longitude', 'dropoff_latitude',
                              count=ds.count('passenger_count'))

        pix = tf.interpolate(pickups.count, (255, 204, 204), 'red')

        print('Dataset: {}, Request: {}'.format(dataset, args))
        t_start = dt.datetime.utcnow()
        pix = tf.interpolate(pickups.count, (255, 204, 204), 'red')
        img_io = pix.to_bytesio()
        self.write(img_io.getvalue())
        self.set_header("Content-type", "image/png")
        t_end = dt.datetime.utcnow()
        elapsed = t_end - t_start
        print('Elapsed time: {}'.format(elapsed.total_seconds()))

class AppState(object):

    def __init__(self):
        self.cache = {}
        self.transfer_functions = []
        self.basemaps = []
        self.datasets = dict([('NYC Taxi Pickups','TAXI_PICKUP'),('GDELT','GDELT'),('US Census Synthetic People','CENSUS_SYN_PEOPLE'),('NYC Taxi Dropoffs','TAXI_DROPOFF')])
        self.basemaps = dict([('Toner','http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png'),
            ('Imagery','http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}'),
            ('Shaded Relief','http://services.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{Z}/{Y}/{X}')])
        self.dataset = self.datasets.keys()[0]
        self.basemap = self.basemaps.keys()[0]
        self.load_datasets()

    def load_datasets(self):
        print('Loading Data...')
        if os.path.exists('data/taxi.castra'):
            df = dd.from_castra('data/taxi.castra')
            df = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
            df = df.compute()
            self.cache['TAXI_PICKUP'] = df

    def bind_widget(self, widget, widget_field, model_field):
        widget.on_change(widget_field, partial(self.on_change, model_field=model_field))

    def on_change(self, attr, old, new, model_field):
        setattr(self, model_field, new)
        self.update_app()

def add_select(self, name, options, model_field):
    '''TODO: add docs'''
    widget = Select.create(name=name, value=getattr(self.model, model_field), options=options)
    self.bind_to_model(widget, 'value', model_field)
    self.layout.children.append(widget)
    return widget

def create_client(model):
    output_file('bokeh_datashader_example.html')
    mercator_extent = dict(start=-20000000, end=20000000, bounds=None)
    x_range = Range1d(**mercator_extent)
    y_range = Range1d(**mercator_extent)
    f = figure(tools='wheel_zoom,pan', x_range=x_range, y_range=y_range)
    f.axis.visible = False
    f.add_tile(STAMEN_TONER)

    service_url = 'http://{host}:{port}/{layer_name}?'
    service_url += 'height={HEIGHT}&'
    service_url += 'width={WIDTH}&'
    service_url += 'arl={arl}&'
    service_url += 'select={XMIN},{YMIN},{XMAX},{YMAX}'

    shader_url_vars = {}
    shader_url_vars['host'] = 'localhost'
    shader_url_vars['port'] = 5000
    shader_url_vars['layer_name'] = model.dataset
    shader_url_vars['arl'] = 'stub'

    dynamic_image_options = {}
    dynamic_image_options['url'] = service_url
    dynamic_image_options['extra_url_vars'] = shader_url_vars
    dynamic_image_source = ImageSource(**dynamic_image_options)

    f.add_dynamic_image(dynamic_image_source)

    # user controls -----------------------------------
    model.dataset_select = Select.create(name='Datasets', value=model.dataset, options=model.datasets)
    
    model.basemap_select = Select.create(name='Basemaps', value=model.basemap, options=model.basemaps)

    controls = VBox(children=[model.dataset_select, model.basemap_select])
    layout = HBox(children=[controls, f])
    show(layout)

def create_server():
    print('Server Starting...')
    server = Server(Application(), io_loop=IOLoop(), extra_patterns=[(r"/(.*)", GetDataset)], port=5000)
    return server

if __name__ == '__main__':
    model = AppState()
    create_client(model)
    server = create_server()
    server.start()
    print('Server Ready...')
