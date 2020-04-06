<img src="https://github.com/holoviz/datashader/raw/master/doc/_static/logo_stacked.png" data-canonical-src="https://github.com/holoviz/datashader/raw/master/doc/_static/logo_stacked.png" width="200"/><br>

-----------------

# Turn even the largest data into images, accurately

|    |    |
| --- | --- |
| Build Status | [![Linux/MacOS Build Status](https://travis-ci.org/holoviz/datashader.svg?branch=master)](https://travis-ci.org/holoviz/datashader) [![Windows Build status](https://img.shields.io/appveyor/ci/holoviz-developers/datashader/master.svg?logo=appveyor)](https://ci.appveyor.com/project/holoviz-developers/datashader/branch/master) |
| Coverage | [![codecov](https://codecov.io/gh/holoviz/datashader/branch/master/graph/badge.svg)](https://codecov.io/gh/holoviz/datashader) |
| Latest dev release | [![Github tag](https://img.shields.io/github/tag/holoviz/datashader.svg?label=tag&colorB=11ccbb)](https://github.com/holoviz/datashader/tags) [![dev-site](https://img.shields.io/website-up-down-green-red/https/pyviz-dev.github.io/datashader.svg?label=dev%20website)](https://pyviz-dev.github.io/datashader/) |
| Latest release | [![Github release](https://img.shields.io/github/release/holoviz/datashader.svg?label=tag&colorB=11ccbb)](https://github.com/holoviz/datashader/releases) [![PyPI version](https://img.shields.io/pypi/v/datashader.svg?colorB=cc77dd)](https://pypi.python.org/pypi/datashader) [![datashader version](https://img.shields.io/conda/v/pyviz/datashader.svg?colorB=4488ff&style=flat)](https://anaconda.org/pyviz/datashader) [![conda-forge version](https://img.shields.io/conda/v/conda-forge/datashader.svg?label=conda%7Cconda-forge&colorB=4488ff)](https://anaconda.org/conda-forge/datashader) [![defaults version](https://img.shields.io/conda/v/anaconda/datashader.svg?label=conda%7Cdefaults&style=flat&colorB=4488ff)](https://anaconda.org/anaconda/datashader) |
| Docs | [![gh-pages](https://img.shields.io/github/last-commit/holoviz/datashader/gh-pages.svg)](https://github.com/holoviz/datashader/tree/gh-pages) [![site](https://img.shields.io/website-up-down-green-red/http/datashader.org.svg)](http://datashader.org) |
| Support | [![Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fdiscourse.holoviz.org)](https://discourse.holoviz.org/) |


## What is it?

Datashader is a data rasterization pipeline for automating the process of
creating meaningful representations of large amounts of data. Datashader
breaks the creation of images of data into 3 main steps:

1. Projection

   Each record is projected into zero or more bins of a nominal plotting grid
   shape, based on a specified glyph.

2. Aggregation

   Reductions are computed for each bin, compressing the potentially large
   dataset into a much smaller *aggregate* array.

3. Transformation

   These aggregates are then further processed, eventually creating an image.

Using this very general pipeline, many interesting data visualizations can be
created in a performant and scalable way. Datashader contains tools for easily
creating these pipelines in a composable manner, using only a few lines of code.
Datashader can be used on its own, but it is also designed to work as
a pre-processing stage in a plotting library, allowing that library
to work with much larger datasets than it would otherwise.

## Installation

Datashader supports Python 2.7, 3.6 and 3.7 on Linux, Windows, or
Mac and can be installed with conda:

    conda install datashader

or with pip:

    pip install datashader

For the best performance, we recommend using conda so that you are sure
to get numerical libraries optimized for your platform. The latest
releases are avalailable on the pyviz channel `conda install -c pyviz
datashader` and the latest pre-release versions are avalailable on the
dev-labelled channel `conda install -c pyviz/label/dev datashader`.

## Fetching Examples

Once you've installed datashader as above you can fetch the examples:

    datashader examples
    cd datashader-examples

This will create a new directory called
<span class="title-ref">datashader-examples</span> with all the data
needed to run the examples.

To run all the examples you will need some extra dependencies. If you
installed datashader **within a conda environment**, with that
environment active run:

    conda env update --file environment.yml

Otherwise create a new environment:

    conda env create --name datashader --file environment.yml
    conda activate datashader

## Developer Instructions

1.  Install Python 3
    [miniconda](https://docs.conda.io/en/latest/miniconda.html) or
    [anaconda](https://www.anaconda.com/distribution/), if you don't
    already have it on your system.

2.  Clone the datashader git repository if you do not already have it:

        git clone git://github.com/holoviz/datashader.git

3.  Set up a new conda environment with all of the dependencies needed
    to run the examples:

        cd datashader
        conda env create --name datashader --file ./examples/environment.yml
        conda activate datashader

4.  Put the datashader directory into the Python path in this
    environment:

        pip install --no-deps -e .

## Learning more

After working through the examples, you can find additional resources linked
from the [datashader documentation](http://datashader.org),
including API documentation and papers and talks about the approach.

## Some Examples

![USA census](examples/assets/images/usa_census.jpg)

![NYC races](examples/assets/images/nyc_races.jpg)

![NYC taxi](examples/assets/images/nyc_pickups_vs_dropoffs.jpg)
