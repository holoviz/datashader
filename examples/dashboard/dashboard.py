from __future__ import absolute_import, print_function, division

import argparse
from os import path
import yaml
import webbrowser
import uuid
import pdb

from collections import OrderedDict

import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import numpy as np
from xarray import DataArray

from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler

from bokeh.plotting import Figure 
from bokeh.models import (Range1d, ImageSource, WMTSTileSource, TileRenderer,
                          DynamicImageRenderer, HBox, VBox)

from bokeh.models import Select, Slider, CheckboxGroup, CustomJS, ColumnDataSource, Square, HoverTool
from bokeh.palettes import BrBG9, PiYG9

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


        # handle categorical field
        if self.model.field in self.model.categorical_fields:
            agg = cvs.points(self.model.df,
                             self.model.active_axes[1],
                             self.model.active_axes[2],
                             ds.count_cat(self.model.field))

            pix = tf.colorize(agg,
                              self.model.colormap,
                              how=self.model.transfer_function)

        # handle ordinal field
        elif self.model.field in self.model.ordinal_fields:
            agg = cvs.points(self.model.df,
                             self.model.active_axes[1],
                             self.model.active_axes[2],
                             self.model.aggregate_function(self.model.field))

            pix = tf.interpolate(agg, cmap=self.model.color_ramp,
                                 how=self.model.transfer_function)
        # handle no field
        else:
            agg = cvs.points(self.model.df,
                             self.model.active_axes[1],
                             self.model.active_axes[2])

            pix = tf.interpolate(agg, cmap=self.model.color_ramp,
                                 how=self.model.transfer_function)

        if self.model.spread_size > 0:
            pix = tf.spread(pix, px=self.model.spread_size)

        def update_plots():

            def downsample_categorical(aggregate, factor):
                ys, xs, zs = aggregate.shape
                crarr = aggregate[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
                return np.nanmean(np.concatenate([[crarr[i::factor,j::factor] 
                                                   for i in range(factor)] 
                                                   for j in range(factor)]), axis=0)

            def downsample(aggregate, factor):
                ys, xs = aggregate.shape
                crarr = aggregate[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
                return np.nanmean(np.concatenate([[crarr[i::factor,j::factor] 
                                                   for i in range(factor)] 
                                                   for j in range(factor)]), axis=0)

            # update hover layer ----------------------------------------------------
            sq_xs = np.linspace(self.model.map_extent[0],
                                self.model.map_extent[2],
                                agg.shape[1] / self.model.hover_size)

            sq_ys = np.linspace(self.model.map_extent[1],
                                self.model.map_extent[3],
                                agg.shape[0] / self.model.hover_size)

            agg_xs, agg_ys = np.meshgrid(sq_xs, sq_ys)
            self.model.hover_source.data['x'] = agg_xs.flatten()
            self.model.hover_source.data['y'] = agg_ys.flatten()

            if self.model.field in self.model.categorical_fields:
                hover_agg = downsample_categorical(agg.values, self.model.hover_size)
                cats = agg[agg.dims[2]].values.tolist()
                tooltips = []
                for i, e in enumerate(cats):
                    self.model.hover_source.data[e] = hover_agg[:,:,i].flatten()
                    tooltips.append((e, '@{}'.format(e)))
                self.model.hover_tool.tooltips = tooltips

            else:
                hover_agg = downsample(agg.values, self.model.hover_size)
                self.model.hover_source.data['value'] = hover_agg.flatten()
                self.model.hover_tool.tooltips = [(self.model.field_title, '@value')]

            # update legend ---------------------------------------------------------
            if self.model.field in self.model.categorical_fields:

                cat_dim = agg.dims[-1]
                len_bar=900
                cats = agg[cat_dim].values.tolist()
                total = agg.sum(dim=cat_dim)
                min_val = int(total.min().data)
                max_val = int(total.max().data)
                scale = np.linspace(min_val, max_val, 180, dtype=total.dtype)
                cats = agg.coords[agg.dims[-1]].values
                ncats = len(cats)
                data = np.zeros((180, ncats, ncats), dtype=total.dtype)
                data[:, np.arange(ncats), np.arange(ncats)] = scale[:, None]
                cbar = DataArray(data, dims=['value', 'fake', cat_dim], 
                                 coords=[scale, cats, cats])
                img = tf.colorize(cbar, self.model.colormap, how=self.model.transfer_function)

                dw = max_val - min_val
                legend_fig = self.model.create_legend(img.values.T,
                                                      x=min_val,
                                                      y=0,
                                                      dh=18 * ncats,
                                                      dw=dw,
                                                      x_start=min_val,
                                                      x_end=max_val,
                                                      y_range=(0,18 * ncats))
                self.model.legend_vbox.children = [legend_fig]

            else:
                min_val = np.nanmin(agg.values)
                max_val = np.nanmax(agg.values)
                vals = np.linspace(min_val, max_val, 180)[None, :]
                vals_arr = DataArray(vals)
                img = tf.interpolate(vals_arr, cmap=self.model.color_ramp,
                                     how=self.model.transfer_function)
                dw = max_val - min_val
                legend_fig = self.model.create_legend(img.values,
                                                      x=min_val,
                                                      y=0,
                                                      dh=18,
                                                      dw=dw,
                                                      x_start=min_val,
                                                      x_end=max_val,
                                                      y_range=(0,18))

                self.model.legend_vbox.children = [legend_fig]

        server.get_sessions('/')[0].with_document_locked(update_plots)
        # serialize to image
        img_io = pix.to_bytesio()
        self.write(img_io.getvalue())
        self.set_header("Content-type", "image/png")

class AppState(object):
    """Simple value object to hold app state"""

    def __init__(self, config_file, outofcore, app_port):

        self.load_config_file(config_file)
        self.plot_height = 560
        self.plot_width = 810

        self.aggregate_functions = OrderedDict()
        self.aggregate_functions['Count'] = ds.count
        self.aggregate_functions['Mean'] = ds.mean
        self.aggregate_functions['Sum'] = ds.sum
        self.aggregate_functions['Min'] = ds.min
        self.aggregate_functions['Max'] = ds.max
        self.aggregate_function = list(self.aggregate_functions.values())[0]

        # transfer function configuration
        self.transfer_functions = OrderedDict()
        self.transfer_functions[u"\u221B - Cube Root"] = 'cbrt'
        self.transfer_functions['Log'] = 'log'
        self.transfer_functions['Linear'] = 'linear'
        self.transfer_functions['Histogram Equalization'] = 'eq_hist'
        self.transfer_function = list(self.transfer_functions.values())[0]

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

        # hover
        self.hover_source = ColumnDataSource(data=dict(x=[], y=[], val=[]))
        self.hover_size = 8

        # spreading
        self.spread_size = 1

        # color ramps
        self.color_ramps = OrderedDict()
        self.color_ramps['BrBG'] = BrBG9
        self.color_ramps['PiYG'] = PiYG9
        self.color_ramp = list(self.color_ramps.values())[0]

    def load_config_file(self, config_path):
        '''load and parse yaml config file'''

        if not path.exists(config_path):
            raise IOError('Unable to find config file "{}"'.format(config_path))

        self.config_path = path.abspath(config_path)

        with open(config_path) as f:
            self.config = yaml.load(f.read())

        # parse initial extent
        extent = self.config['initial_extent']
        self.map_extent = [extent['xmin'], extent['ymin'],
                           extent['xmax'], extent['ymax']]

        # parse plots
        self.axes = OrderedDict()
        for p in self.config['axes']:
            self.axes[p['name']] = (p['name'], p['xaxis'], p['yaxis'])
        self.active_axes = list(self.axes.values())[0]

        # parse summary field
        self.fields = OrderedDict()
        self.colormaps = OrderedDict()
        self.ordinal_fields = []
        self.categorical_fields = []
        for f in self.config['summary_fields']:
            self.fields[f['name']] = None if f['field'] == 'None' else f['field']

            if 'cat_colors' in f.keys():
                self.colormaps[f['name']] = f['cat_colors']
                self.categorical_fields.append(f['field'])

            elif f['field'] != 'None':
                self.ordinal_fields.append(f['field'])

        self.field = list(self.fields.values())[0]
        self.field_title = list(self.fields.keys())[0]

        if self.colormaps:
            self.colormap = self.colormaps[list(self.fields.keys())[0]]

    def load_datasets(self,outofcore):
        data_path = self.config['file']
        print('Loading Data from {}...'.format(data_path))

        if not path.isabs(data_path):
            config_dir = path.split(self.config_path)[0]
            data_path = path.join(config_dir, data_path)

        if not path.exists(data_path):
            raise IOError('Unable to find input dataset: "{}"'.format(data_path))

        axes_fields = []
        for f in self.axes.values():
            axes_fields += [f[1], f[2]]

        load_fields = [f for f in self.fields.values() if f is not None] + axes_fields

        if data_path.endswith(".csv"):
            self.df = pd.read_csv(data_path, usecols=load_fields)

            # parse categorical fields
            for f in self.categorical_fields:
                self.df[f] = self.df[f].astype('category')

        elif data_path.endswith(".castra"):
            import dask.dataframe as dd
            self.df = dd.from_castra(data_path)
            if not outofcore:
                self.df = self.df.cache(cache=dict)
            
        else:
            raise IOError("Unknown data file type; .csv and .castra currently supported")

    def create_legend(self, img, x, y, dw, dh, x_start, x_end, y_range):
        legend_fig = Figure(x_range=(x_start, x_end),
                            plot_height=max(dh, 50),
                            plot_width=self.plot_width,
                            lod_threshold=None,
                            toolbar_location=None,
                            y_range=y_range)

        legend_fig.min_border_top = 0
        legend_fig.min_border_bottom = 10
        legend_fig.min_border_left = 0
        legend_fig.min_border_right = 0
        legend_fig.yaxis.visible = False
        legend_fig.grid.grid_line_alpha = 0

        legend_fig.image_rgba(image=[img],
                              x=[x],
                              y=[y],
                              dw=[dw],
                              dh=[dh],
                              dw_units='screen')
        return legend_fig

        
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
                          y_range=self.y_range)

        self.fig.min_border_top = 0
        self.fig.min_border_bottom = 10
        self.fig.min_border_left = 0
        self.fig.min_border_right = 0
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
        
        # add label layer
        self.label_source = WMTSTileSource(url=self.model.labels_url)
        self.label_renderer = TileRenderer(tile_source=self.label_source)
        self.fig.renderers.append(self.label_renderer)

        # Add a hover tool
        self.invisible_square = Square(x='x',
                                       y='y',
                                       fill_color=None,
                                       line_color=None, 
                                       size=self.model.hover_size)

        self.visible_square = Square(x='x',
                                     y='y', 
                                     fill_color='#79DCDE',
                                     fill_alpha=.5,
                                     line_color='#79DCDE', 
                                     line_alpha=1,
                                     size=self.model.hover_size)

        cr = self.fig.add_glyph(self.model.hover_source,
                                self.invisible_square,
                                selection_glyph=self.visible_square,
                                nonselection_glyph=self.invisible_square)

        code = "source.set('selected', cb_data['index']);"
        callback = CustomJS(args={'source': self.model.hover_source}, code=code)
        self.model.hover_tool = HoverTool(tooltips=[(self.model.fields.keys()[0], "@value")],
                                    callback=callback, 
                                    renderers=[cr], 
                                    mode='mouse')
        self.fig.add_tools(self.model.hover_tool)
        self.model.legend_vbox = VBox()

        # add ui components
        controls = []
        axes_select = Select.create(name='Axes',
                                    options=self.model.axes)
        axes_select.on_change('value', self.on_axes_change)
        controls.append(axes_select)

        self.field_select = Select.create(name='Field', options=self.model.fields)
        self.field_select.on_change('value', self.on_field_change)
        controls.append(self.field_select)

        self.aggregate_select = Select.create(name='Aggregate',
                                         options=self.model.aggregate_functions)
        self.aggregate_select.on_change('value', self.on_aggregate_change)
        controls.append(self.aggregate_select)

        transfer_select = Select.create(name='Transfer Function',
                                        options=self.model.transfer_functions)
        transfer_select.on_change('value', self.on_transfer_function_change)
        controls.append(transfer_select)

        color_ramp_select = Select.create(name='Color Ramp', options=self.model.color_ramps)
        color_ramp_select.on_change('value', self.on_color_ramp_change)
        controls.append(color_ramp_select)

        # add map components
        basemap_select = Select.create(name='Basemap', value='Toner',
                                       options=self.model.basemaps)
        basemap_select.on_change('value', self.on_basemap_change)

        image_opacity_slider = Slider(title="Opacity", value=100, start=0,
                                      end=100, step=1)
        image_opacity_slider.on_change('value', self.on_image_opacity_slider_change)

        basemap_opacity_slider = Slider(title="Basemap Opacity", value=100, start=0,
                                        end=100, step=1)
        basemap_opacity_slider.on_change('value', self.on_basemap_opacity_slider_change)

        spread_size_slider = Slider(title="Spread Size (px)", value=0, start=0,
                                        end=10, step=1)
        spread_size_slider.on_change('value', self.on_spread_size_change)
        controls.append(spread_size_slider)

        hover_size_slider = Slider(title="Hover Size (px)", value=8, start=4,
                                        end=30, step=1)
        hover_size_slider.on_change('value', self.on_hover_size_change)
        controls.append(hover_size_slider)


        show_labels_chk = CheckboxGroup(labels=["Show Labels"], active=[0])
        show_labels_chk.on_click(self.on_labels_change)

        map_controls = [basemap_select, basemap_opacity_slider,
                        image_opacity_slider, show_labels_chk]

        self.controls = VBox(width=200, height=600, children=controls)
        self.map_controls = HBox(width=self.fig.plot_width, children=map_controls)
        self.map_area = VBox(width=self.fig.plot_width, children=[self.map_controls,
                                                                  self.fig,
                                                                  self.model.legend_vbox])
        self.layout = HBox(width=1024, children=[self.controls, self.map_area])

    def update_image(self):
        self.model.shader_url_vars['cachebust'] = str(uuid.uuid4())
        self.image_renderer.image_source = ImageSource(url=self.model.service_url,
                        extra_url_vars=self.model.shader_url_vars)

    def on_field_change(self, attr, old, new):
        self.model.field_title = new
        self.model.field = self.model.fields[new]
        self.update_image()

        if not self.model.field:
            self.aggregate_select.options = [dict(name="No Aggregates Available", value="")]
        elif self.model.field in self.model.categorical_fields:
            self.aggregate_select.options = [dict(name="Categorical", value="count_cat")]
        else:
            opts = [dict(name=k, value=k) for k in self.model.aggregate_functions.keys()]
            self.aggregate_select.options = opts

    def on_basemap_change(self, attr, old, new):
        self.model.basemap = self.model.basemaps[new]
        self.tile_renderer.tile_source = WMTSTileSource(url=self.model.basemap)

    def on_hover_size_change(self, attr, old, new):
        self.model.hover_size = int(new)
        self.invisible_square.size = int(new)
        self.visible_square.size = int(new)
        self.update_image()

    def on_spread_size_change(self, attr, old, new):
        self.model.spread_size = int(new)
        self.update_image()

    def on_axes_change(self, attr, old, new):
        self.model.active_axes = self.model.axes[new]
        self.update_image()

    def on_aggregate_change(self, attr, old, new):
        self.model.aggregate_function = self.model.aggregate_functions[new]
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
    parser.add_argument('-p', '--port',   help='port number to use for communicating with server; defaults to 5000', default=5000)
    parser.add_argument('-o', '--outofcore', help='use out-of-core processing if available, for datasets larger than memory',
                        default=False, action='store_true')
    args = vars(parser.parse_args())

    APP_PORT = args['port']

    def add_roots(doc):
        model = AppState(args['config'], args['outofcore'], APP_PORT)
        view = AppView(model)
        GetDataset.model = model
        doc.add_root(view.layout)

    app = Application()
    app.add(FunctionHandler(add_roots))
    # Start server object wired to bokeh client. Instantiating ``Server``
    # directly is used to add custom http endpoint into ``extra_patterns``.
    server = Server(app, io_loop=IOLoop(),
                    extra_patterns=[(r"/datashader", GetDataset)], port=APP_PORT)

    print('Starting server at http://localhost:{}/...'.format(APP_PORT))
    webbrowser.open('http://localhost:{}'.format(APP_PORT))

    server.start()
