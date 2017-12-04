import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf

from bokeh.plotting import figure, Document
from datashader.bokeh_ext import InteractiveImage

axis = ds.core.LinearAxis()
lincoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 2), 2)
coords = [lincoords, lincoords]
dims = ['y', 'x']

df = pd.DataFrame({'x': np.array(([0.] * 10 + [1] * 10)),
                   'y': np.array(([0.] * 5 + [1] * 5 + [0] * 5 + [1] * 5))})

def create_image(x_range, y_range, plot_width=2, plot_height=2):
    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                    x_range=x_range, y_range=y_range)
    img = cvs.points(df, 'x', 'y')
    return tf.shade(img)


def test_interactive_image_initialize():
    p = figure(x_range=(0, 1), y_range=(0, 1), plot_width=2, plot_height=2)
    img = InteractiveImage(p, create_image)
    out = np.array([[4287299584, 4287299584],
                    [4287299584, 4287299584]], dtype=np.uint32)

    assert img.ds.data['x'] == [0]
    assert img.ds.data['y'] == [0]
    assert img.ds.data['dh'] == [1]
    assert img.ds.data['dw'] == [1]
    assert np.array_equal(img.ds.data['image'][0], out)

    assert img.renderer.glyph.x == 'x'
    assert img.renderer.glyph.y == 'y'
    assert img.renderer.glyph.dh == 'dh'
    assert img.renderer.glyph.dw == 'dw'
    assert img.renderer.glyph.image == 'image'


def test_interactive_image_update():
    p = figure(x_range=(0, 1), y_range=(0, 1), plot_width=2, plot_height=2)
    img = InteractiveImage(p, create_image)

    # Ensure bokeh Document is instantiated
    img._repr_html_()
    assert isinstance(img.doc, Document) 

    # Ensure image is updated
    img.update_image({'xmin': 0.5, 'xmax': 1, 'ymin': 0.5, 'ymax': 1, 'w': 1, 'h': 1})
    out = np.array([[4287299584]], dtype=np.uint32)
    assert img.ds.data['x'] == [0.5]
    assert img.ds.data['y'] == [0.5]
    assert img.ds.data['dh'] == [0.5]
    assert img.ds.data['dw'] == [0.5]
    assert np.array_equal(img.ds.data['image'][0], out)

    # Ensure patch message is correct
    msg = img.get_update_event()
    event = msg.content['events'][0]
    assert event['kind'] == 'ColumnDataChanged'
    assert event['column_source'] == img.ds.ref
    assert sorted(event['cols']) == ['dh', 'dw', 'image', 'x', 'y']
    new = event['new']
    assert new['dh'] == [0.5]
    assert new['dw'] == [0.5]
    assert new['x'] == [0.5]
    assert new['y'] == [0.5]
    image = new['image'][0]
    assert image['dtype'] == 'uint32'
    assert image['shape'] == [1, 1]

    # Ensure events are cleared after update
    assert img.doc._held_events == []
