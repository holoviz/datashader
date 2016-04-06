Datashader
==========

Datashader is a graphics pipeline system for creating meaningful
representations of large amounts of data. It breaks the creation of images into
3 steps:

1. Projection

   Each record is projected into zero or more bins, based on a specified glyph.

2. Aggregation

   Reductions are computed for each bin, compressing the potentially large
   dataset into a much smaller *aggregate*.

3. Transformation

   These aggregates are then further processed to create an image.

Using this very general pipeline, many interesting data visualizations can be
created in a performant and scalable way. Datashader contains tools for easily
creating these pipelines in a composable manner, using only a few lines of code:


.. code-block:: python

    >>> import datashader as ds
    >>> import datashader.transfer_functions as tf

    >>> import pandas as pd
    >>> df = pd.read_csv('user_data.csv')

    # **Projection & Aggregation Step:**
    # Map each record as a point centered by the fields `x_col` and `y_col` to
    # a 400x400 grid of bins, computing the mean of `z_col` for all records in
    # each bin.
    >>> cvs = ds.Canvas(plot_width=400, plot_height=400)
    >>> agg = cvs.points(df, 'x_col', 'y_col', ds.mean('z_col'))

    # **Transformation Step:**
    # Interpolate the resulting means along a logarithmic color palette from
    # "lightblue" to "darkblue"
    >>> img = tf.interpolate(agg, cmap=['lightblue', 'darkblue'], how='log')


Examples
--------

The repository contains several runnable examples, which can be `found here
<https://github.com/bokeh/datashader/tree/master/examples>`_. Many of the
examples are in the form of Jupyter notebooks. Copies of these with all the
images and output included can be viewed on `Anaconda Cloud
<https://anaconda.org/jbednar/notebooks>`_.


Other resources
---------------

`Video <http://go2.continuum.io/JN12XH0g0W0Rb300CZ00000>`_ and
`slides <http://go2.continuum.io/V0Nc000C300W100X20HZhR0>`_ from a Feb
2015 one-hour talk introducing Datashader are available.

Some of the original ideas for datashader were developed under the
name Abstract Rendering, which is described in a `2014 SPIE VDA paper
<http://www.crest.iu.edu/publications/prints/2014/Cottam2014OutOfCore.pdf>`_.


.. toctree::
   :maxdepth: 1
   :hidden:

   api.rst
