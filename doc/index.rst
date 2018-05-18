.. image:: _static/logo_stacked.png
   :height: 220px
   :align: left

.. raw:: html

   <p style="font-size:20px"><b>Turns even the largest data into images, accurately.</b></p>

Datashader is a graphics pipeline system for creating meaningful
representations of large datasets quickly and flexibly. Datashader
breaks the creation of images into a series of explicit steps that
allow computations to be done on intermediate representations.  This
approach allows accurate and effective visualizations to be produced
automatically without trial-and-error parameter tuning, and also makes
it simple for data scientists to focus on particular data and
relationships of interest in a principled way.

.. raw:: html
  :file: latest_news.html

The computation-intensive steps in this process are written in Python
but transparently compiled to machine code using `Numba`_ and flexibly
distributed across cores and processors using `Dask`_, providing a
highly optimized rendering pipeline that makes it practical to work
with extremely large datasets even on standard hardware.

.. _`Dask`: http://dask.pydata.org
.. _`Numba`: http://numba.pydata.org
.. _`PyViz.org`: http://pyviz.org

To make it concrete, here's an example of what datashader code looks like:

.. code-block:: python

    >>> import datashader as ds
    >>> import datashader.transfer_functions as tf
    >>> import pandas as pd
    >>> df = pd.read_csv('user_data.csv')

    >>> cvs = ds.Canvas(plot_width=400, plot_height=400)
    >>> agg = cvs.points(df, 'x_col', 'y_col', ds.mean('z_col'))
    >>> img = tf.shade(agg, cmap=['lightblue', 'darkblue'], how='log')

This code reads a data file into a Pandas dataframe ``df``, and then
projects the fields ``x_col`` and ``y_col`` onto the x and y dimensions of
400x400 grid, aggregating it by the mean value of the ``z_col`` of each
datapoint. The results are rendered into an image where the minimum
count will be plotted in ``lightblue``, the maximum in ``darkblue``, and
ranging logarithmically in between.

And here are some sample outputs for 300 million points of data (one
per person in the USA) from the 2010 census, each constructed using
code like the above:

.. image:: assets/images/usa_census.jpg

.. image:: assets/images/nyc_races.jpg

           
Installation
------------

Please follow the instructions on the `Github repo <https://github.com/bokeh/datashader/tree/master/examples>`_
if you want to reproduce the specific examples on this website, or the ones at `PyViz.org <http://pyviz.org>`_ if you want
to try out Datashader together with related plotting tools.



Other resources
---------------

You can watch a short talk about datashader on YouTube:
`Datashader: Revealing the Structure of Genuinely Big Data`_.
The video `Visualizing Billions of Points of Data`_ (and its `slides`_)
from a February 2016 one-hour talk first introducing Datashader are also
available, but do not cover more recent extensions to the library.

.. _`Datashader: Revealing the Structure of Genuinely Big Data`: https://www.youtube.com/watch?v=6m3CFbKmK_c
.. _`Visualizing Billions of Points of Data`: http://go2.continuum.io/JN12XH0g0W0Rb300CZ00000
.. _`slides`: http://go2.continuum.io/V0Nc000C300W100X20HZhR0

Some of the original ideas for datashader were developed under the
name Abstract Rendering, which is described in a `2014 SPIE VDA paper`_.

.. _`2014 SPIE VDA paper`: http://www.crest.iu.edu/publications/prints/2014/Cottam2014OutOfCore.pdf

The source code for datashader is maintained at our `Github site,`_ and
is documented using the API link on this page.

.. _`GitHub site,`: https://github.com/bokeh/datashader

We recommend the `Getting Started Guide <getting_started>`_ to learn
the basic concepts and start using Datashader as quickly as possible.

The `User Guide <user_guide>`_ covers specific topics in more detail.

The `API <Reference_Manual>`_ is the definitive guide to each part of
Datashader, but the same information is available more conveniently via
the `help()` command as needed when using each component.

Please feel free to report `issues
<https://github.com/ioam/holoviews/issues>`_ or `contribute code.
<https://help.github.com/articles/about-pull-requests>`_ You are also
welcome to chat with the developers on `gitter.
<https://gitter.im/ioam/holoviews>`_


.. toctree::
   :hidden:
   :maxdepth: 2

   Introduction <self>
   Getting Started <getting_started/index>
   User Guide <user_guide/index>
   Topics <topics/index>
   API <api>
   FAQ

