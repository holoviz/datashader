***************
Getting Started
***************

Installation
------------

Datashader supports Python 3.11, 3.12, 3.13, 3.14 and 3.15 on Linux, Windows, or Mac
and can be installed with conda::

    conda install datashader

or with pip::

    pip install datashader

For the best performance, we recommend using conda so that you are
sure to get numerical libraries optimized for your platform.
The latest releases are available on the pyviz channel ``conda install -c pyviz datashader``
and the latest pre-release versions are available on the dev-labeled channel
``conda install -c pyviz/label/dev datashader``.

Fetching Examples
-----------------

The examples are the Jupyter notebooks in the ``examples`` folder of the
Datashader repository. To get them, clone the repository::

    git clone https://github.com/holoviz/datashader.git
    cd datashader/examples

The notebooks create or download the data they need when you run them.

To run all the examples you will need some extra dependencies. If you installed
datashader **within a conda environment**, add them to that environment,
replacing ``<env-name>`` with its name::

    conda env update --name <env-name> --file environment.yml

Otherwise create a new environment::

    conda env create --name datashader --file environment.yml
    conda activate datashader

Usage
-----

.. notebook:: datashader ../../examples/getting_started/index.ipynb
    :offset: 0

.. toctree::
    :titlesonly:
    :maxdepth: 2
    :hidden:

    Introduction <Introduction>
    Pipeline <Pipeline>
    Interactivity <Interactivity>

If you have any questions, please refer to `FAQ <../FAQ>`_
and if that doesn't help, feel free to post an
`issue on GitHub <https://github.com/holoviz/datashader/issues>`_ or a
`question on discourse <https://discourse.holoviz.org/c/datashader/>`_.

Developer Instructions
----------------------

1. Install Python 3 `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `anaconda <https://www.anaconda.com/distribution/>`_, if you don't already have it on your system.

2. Clone the datashader git repository if you do not already have it::

    git clone https://github.com/holoviz/datashader.git

3. Set up a new conda environment with all of the dependencies needed to run the examples::

    cd datashader
    conda env create --name datashader --file ./examples/environment.yml
    conda activate datashader

4. Put the datashader directory into the Python path in this environment::

    pip install --no-deps -e .
