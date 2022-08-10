.. Datashader documentation master file

.. raw:: html

  <h1><img src="_static/logo_horizontal.png" style="width: 50%;"></h1>

**Accurately render even the largest data**

.. raw:: html

  <div style="display: flex">
    <div style="width: 70%">


New to Datashader? Check out this
`quick video introduction to what it does and how it works <https://youtu.be/U6dyIRolIhI>`_!

Datashader is a graphics pipeline system for creating meaningful representations of large
datasets quickly and flexibly. Datashader breaks the creation of images into a series of explicit
steps that allow computations to be done on intermediate representations. This approach allows
accurate and effective visualizations to be produced automatically without trial-and-error parameter
tuning, and also makes it simple for data scientists to focus on particular data and relationships
of interest in a principled way.

The computation-intensive steps in this process are written in ordinary Python
but transparently compiled to machine code using `Numba <https://numba.pydata.org>`_  and flexibly
distributed across CPU cores and processors using `Dask <https://dask.pydata.org>`_ or GPUs
using `CUDA <https://github.com/rapidsai/cudf>`_. This approach provides a highly optimized
rendering pipeline that makes it practical to work with extremely large datasets even on standard
hardware, while exploiting distributed and GPU systems when available.

.. raw:: html

  </div>

.. raw:: html
    :file: latest_news.html

.. raw:: html

  </div>
  <hr width='100%'></hr>

.. notebook:: datashader ../examples/index.ipynb
    :offset: 0
    :disable_interactivity_warning:

.. toctree::
    :hidden:
    :maxdepth: 3

    Introduction <self>
    Getting Started <getting_started/index>
    User Guide <user_guide/index>
    Topics <topics/index>
    Releases <releases>
    API <api>
    FAQ <FAQ>
    About <about>
