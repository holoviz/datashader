Datashader
==========

Datashader is a graphics pipeline system for creating meaningful
representations of large amounts of data. It breaks the creation of images into
3 steps:

1. Projection

   Each record is projected into zero or more bins, based on a specified glyph.

2. Aggregation

   Reductions are computed for each bin, compressing the potentially
   large dataset into a much smaller *aggregate*.  In practice,
   aggregation is typically done incrementally, at the same time as
   projection, and so it is not usually necessary to hold the entire
   dataset in memory, but conceptually this is a separate step.

3. Transformation

   These aggregates are then further processed, typically to create an
   image for display but also allowing statistics and other summary
   information to be computed and displayed.

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


FAQ
---

**Q:** When should I use datashader?

**A:** Datashader is designed for working with large datasets, for
cases where it is most crucial to faithfully represent the
*distribution* of your data.  datashader can work easily with
extremely large datasets, with only a fixed-size data structure
(regardless of the original number of records) being transferred to
your local browser for display.  If you ever find yourself subsampling
your data just so that you can plot it feasibly, or if you are forced
for practical reasons to iterate over chunks of it rather than looking
at all of it at once, then datashader can probably help you.


**Q:** When should I *not* use datashader?

**A:** If you have a very small number of data points (in the hundreds
or thousands) or curves (in the tens or several tens, each with
hundreds or thousands of points), then conventional plotting packages
like `bokeh <https://bokeh.pydata.org>`_ may be more suitable.  With
conventional browser-based packages, all of the data points are passed
directly to the browser for display, allowing specific interaction
with each curve or point, including display of metadata, linking to
sources, etc.  This approach offers the most flexibility *per point*
or *per curve*, but rapidly runs into limitations on how much data can
be processed by the browser, and how much can be displayed on screen
and resolved by the human visual system.  If you are not having such
problems, i.e., your data is easily handled by your plotting
infrastructure and you can easily see and work with all your data
onscreen already, then you probably don't need datashader.


**Q:** Is datashader part of bokeh?

**A:** datashader is an independent project, focusing on generating
aggregate arrays and representations of them as images.  Bokeh is a
complementary project, focusing on building browser-based
visualizations and dashboards.  Bokeh (along with other plotting
packages) can display images rendered by datashader, providing axes,
interactive zooming and panning, selection, legends, hover
information, and so on.  Sample bokeh-based plotting code is provided
with datashader, but similar code could be developed for any other
plotting package that can display images, and the library can also be
used separately, without any external plotting packages, generating
images that can be displayed directly or saved to disk, or generating
aggregate arrays suitable for further analysis.


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
