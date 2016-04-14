import math

from bokeh.io import curdoc
from bokeh.plotting import Figure
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.tile_providers import STAMEN_TONER
from bokeh.models import VBox, HBox, Paragraph
from bokeh.palettes import BuGn9

import pandas as pd

from pdb import set_trace

import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd

# load nyc taxi data -------------------------------
path = './data/nyc_taxi.csv'
datetime_field = 'tpep_dropoff_datetime'
cols = ['dropoff_x', 'dropoff_y', 'trip_distance', datetime_field]
df = pd.read_csv(path, usecols=cols, parse_dates=[datetime_field]).dropna(axis=0)
times = pd.DatetimeIndex(df[datetime_field])
grouped = df.groupby([times.hour])
group_count = len(grouped)

# manage client-side dimensions --------------------
def update_dims(attr, old, new):
    pass

dims = ColumnDataSource(data=dict(width=[], height=[], xmin=[], xmax=[], ymin=[], ymax=[]))
dims.on_change('data', update_dims)

dims_jscode = """

var update_dims = function () {
    var new_data = {};
    new_data['height'] = [plot.get('frame').get('height')];
    new_data['width'] = [plot.get('frame').get('width')];
    new_data['xmin'] = [plot.get('x_range').get('start')];
    new_data['ymin'] = [plot.get('y_range').get('start')];
    new_data['xmax'] = [plot.get('x_range').get('end')];
    new_data['ymax'] = [plot.get('y_range').get('end')];
    dims.set('data', new_data);
};

if (typeof throttle != 'undefined' && throttle != null) {
    clearTimeout(throttle);
}

throttle = setTimeout(update_dims, 200, "replace");
"""


# create plot -------------------------------
xmin = -8240227.037
ymin = 4974203.152
xmax = -8231283.905
ymax = 4979238.441
fig = Figure(x_range=(xmin, xmax), y_range=(ymin, ymax), plot_height=600, plot_width=900)
fig.background_fill_color = 'black'
fig.add_tile(STAMEN_TONER, alpha=.3)
fig.x_range.callback = CustomJS(code=dims_jscode, args=dict(plot=fig, dims=dims))
fig.y_range.callback = CustomJS(code=dims_jscode, args=dict(plot=fig, dims=dims))
fig.axis.visible = False
fig.grid.grid_line_alpha = 0

image_source = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
fig.image_rgba(source=image_source, image='image', x='x', y='y', dw='dw', dh='dh', dilate=False)

def update_image(dataframe):
    global dims
    dims_data = dims.data

    if not dims_data['width'] or not dims_data['height']:
        return

    plot_width = int(math.ceil(dims_data['width'][0]))
    plot_height = int(math.ceil(dims_data['height'][0]))
    x_range = (dims_data['xmin'][0], dims_data['xmax'][0])
    y_range = (dims_data['ymin'][0], dims_data['ymax'][0])

    canvas = ds.Canvas(plot_width=plot_width,
                       plot_height=plot_height,
                       x_range=x_range,
                       y_range=y_range)

    agg = canvas.points(dataframe, 'dropoff_x', 'dropoff_y', ds.count('trip_distance'))
    img = tf.interpolate(agg, cmap=BuGn9, how='log')

    new_data = {}
    new_data['image'] = [img.data]
    new_data['x'] = [x_range[0]]
    new_data['y'] = [y_range[0]]
    new_data['dh'] = [y_range[1] - y_range[0]]
    new_data['dw'] = [x_range[1] - x_range[0]]

    image_source.stream(new_data, 1)

time_text = Paragraph(text='Time Period')
controls = HBox(children=[time_text])
layout = VBox(children=[fig, controls])

counter = 0
def update_data():
    global dims, grouped, group_count, counter, time_text

    dims_data = dims.data

    if not dims_data['width'] or not dims_data['height']:
        return

    group = counter % group_count
    update_image(grouped.get_group(group))
    time_text.text = 'Time Period: {}:00 - {}:00'.format(str(group).zfill(2), str(group +1).zfill(2))
    counter += 1

curdoc().add_root(layout)
curdoc().add_periodic_callback(update_data, 1000)
