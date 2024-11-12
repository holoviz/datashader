from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import datashader as ds
import datashader.transfer_functions as tf


df = pd.DataFrame({'x': np.array([0.] * 10 + [1] * 10),
                   'y': np.array([0.] * 5 + [1] * 5 + [0] * 5 + [1] * 5),
                   'f64': np.arange(20, dtype='f8')})
df.loc['f64', 2] = np.nan

cvs = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))
cvs10 = ds.Canvas(plot_width=10, plot_height=10, x_range=(0, 1), y_range=(0, 1))


def test_pipeline():
    pipeline = ds.Pipeline(df, ds.Point('x', 'y'))
    img = pipeline((0, 1), (0, 1), 2, 2)
    agg = cvs.points(df, 'x', 'y', ds.count())
    assert img.equals(tf.shade(agg))

    def color_fn(agg):
        return tf.shade(agg, 'pink', 'red')
    pipeline.color_fn = color_fn
    img = pipeline((0, 1), (0, 1), 2, 2)
    assert img.equals(color_fn(agg))

    def transform_fn(agg):
        return agg + 1
    pipeline.transform_fn = transform_fn
    img = pipeline((0, 1), (0, 1), 2, 2)
    assert img.equals(color_fn(transform_fn(agg)))

    pipeline = ds.Pipeline(df, ds.Point('x', 'y'), ds.sum('f64'))
    img = pipeline((0, 1), (0, 1), 2, 2)
    agg = cvs.points(df, 'x', 'y', ds.sum('f64'))
    assert img.equals(tf.shade(agg))


@pytest.mark.parametrize("line_width", [0.0, 0.5, 1.0, 2.0])
def test_pipeline_antialias(line_width):
    glyph = ds.glyphs.LineAxis0('x', 'y')

    glyph.set_line_width(line_width=line_width)
    assert glyph._line_width == line_width
    assert glyph.antialiased == (line_width > 0)

    pipeline = ds.Pipeline(df, glyph)
    img = pipeline(width=cvs10.plot_width, height=cvs10.plot_height,
                   x_range=cvs10.x_range, y_range=cvs10.y_range)
    agg = cvs10.line(df, 'x', 'y', agg=ds.reductions.count(), line_width=line_width)
    assert img.equals(tf.dynspread(tf.shade(agg)))
