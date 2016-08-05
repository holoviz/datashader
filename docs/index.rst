Datashader
==========

Datashader is a graphics pipeline system for creating meaningful
representations of large datasets quickly and flexibly. Datashader
breaks the creation of images into a series of explicit steps that
allow computations to be done on intermediate representations.  This
approach allows accurate and effective visualizations to be produced
automatically, and also makes it simple for data scientists to focus
on particular data and relationships of interest in a principled way.
Using highly optimized rendering routines written in Python but
compiled to machine code using `Numba <http://numba.pydata.org>`_,
datashader makes it practical to work with extremely large datasets
even on standard hardware.

To make it concrete, here's an example of what datashader code looks like:

.. code-block:: python

    >>> import datashader as ds
    >>> import datashader.transfer_functions as tf
    >>> import pandas as pd
    >>> df = pd.read_csv('user_data.csv')

    >>> cvs = ds.Canvas(plot_width=400, plot_height=400)
    >>> agg = cvs.points(df, 'x_col', 'y_col', ds.mean('z_col'))
    >>> img = tf.interpolate(agg, cmap=['lightblue', 'darkblue'], how='log')

This code reads a data file into a Pandas dataframe ``df``, and then
projects the fields ``x_col`` and ``y_col`` onto the x and y dimensions of
400x400 grid, aggregating it by the mean value of the ``z_col`` of each
datapoint. The results are rendered into an image where the minimum
count will be plotted in ``lightblue``, the maximum in ``darkblue``, and
ranging logarithmically in between.

And here are some sample outputs for data from the 2010 US census,
each constructed using a similar set of code:

.. image:: https://raw.githubusercontent.com/bokeh/datashader/master/docs/images/usa_census.jpg

.. image:: https://raw.githubusercontent.com/bokeh/datashader/master/docs/images/nyc_races.jpg


Documentation for datashader is primarily provided in the form of
Jupyter notebooks.  To understand which plotting problems datashader
helps you avoid, you can start with our `plotting pitfalls notebook
<https://anaconda.org/jbednar/plotting_pitfalls/notebook>`_.  To see the steps
in the datashader pipeline in detail, you can start with our `pipeline
notebook <https://anaconda.org/jbednar/pipeline/notebook>`_.  Or you
may want to start with detailed case studies of datashader in action,
such as our
`NYC Taxi notebook <https://anaconda.org/jbednar/nyc_taxi/notebook>`_ and 
`US Census notebook <https://anaconda.org/jbednar/census/notebook>`_.
Additional notebooks showing how to use datashader for
other applications or data types are viewable on `Anaconda Cloud
<https://anaconda.org/jbednar/notebooks>`_ and can be downloaded
in runnable form at our `Github site
<https://github.com/bokeh/datashader/tree/master/examples>`_
    

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

You can watch a short talk about datashader on
`YouTube <https://www.youtube.com/watch?v=6m3CFbKmK_c>`_.  
`Video <http://go2.continuum.io/JN12XH0g0W0Rb300CZ00000>`_ and
`slides <http://go2.continuum.io/V0Nc000C300W100X20HZhR0>`_ from a Feb
2016 one-hour talk introducing Datashader are also available, but
do not include recent extensions to the library.

Some of the original ideas for datashader were developed under the
name Abstract Rendering, which is described in a `2014 SPIE VDA paper
<http://www.crest.iu.edu/publications/prints/2014/Cottam2014OutOfCore.pdf>`_.

The source code for datashader is maintained at our `Github site
<https://github.com/bokeh/datashader>`_, and is documented using the
API link on this page.

.. toctree::
   :maxdepth: 1
   :hidden:

   api.rst
