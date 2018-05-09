Datashader
----------

[![Travis build Status](https://travis-ci.org/bokeh/datashader.svg?branch=master)](https://travis-ci.org/bokeh/datashader)
[![Appveyor build status](https://ci.appveyor.com/api/projects/status/h3lwh6ju4hfcgkm8/branch/master?svg=true)](https://ci.appveyor.com/project/bokeh-integrations/datashader/branch/master)
[![Task Status](https://badge.waffle.io/bokeh/datashader.png?label=ready&title=tasks)](https://waffle.io/bokeh/datashader)


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

The best way to get started with Datashader is install it together
with our extensive set of examples, following the instructions in the
[examples README](/examples/README.md).

If all you need is datashader itself, without any of the files used in
the examples, you can install it from the `bokeh` channel using the using the
[`conda` package manager](https://conda.io/docs/install/quick.html):

```bash
conda install -c bokeh datashader
```

If you want to get the very latest unreleased changes to datashader
(e.g. to edit the source code yourself), first install using conda 
as above to ensure the dependencies are installed, and you can
then tell Python to use a git clone instead:

```bash
conda remove --force datashader
git clone https://github.com/bokeh/datashader.git
cd datashader
pip install -e .
```

Datashader is not currently available on PyPI, to avoid broken or
low-performance installations that come from not keeping track of
C/C++ binary dependencies such as LLVM (required by Numba).

To run the test suite, first install pytest (e.g. ``conda install
pytest``), then run ``py.test datashader`` in your datashader source
directory.

## Learning more

After working through the examples, you can find additional resources linked
from the [datashader documentation](http://datashader.org),
including API documentation and papers and talks about the approach.

## Screenshots

![USA census](examples/assets/images/usa_census.jpg)

![NYC races](examples/assets/images/nyc_races.jpg)

![NYC taxi](examples/assets/images/nyc_pickups_vs_dropoffs.jpg)
