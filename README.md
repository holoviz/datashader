Datashader
----------

[![Build Status](https://travis-ci.org/bokeh/datashader.svg)](https://travis-ci.org/bokeh/datashader)
[![Documentation Status](https://readthedocs.org/projects/datashader/badge/?version=latest)](http://datashader.readthedocs.org/en/latest/?badge=latest)
[![Task Status](https://badge.waffle.io/bokeh/datashader.png?label=ready&title=tasks)](https://waffle.io/bokeh/datashader)


Datashader is a graphics pipeline system for creating meaningful
representations of large amounts of data. It breaks the creation of images into
3 main steps:

1. Projection

   Each record is projected into zero or more bins, based on a specified glyph.

2. Aggregation

   Reductions are computed for each bin, compressing the potentially large
   dataset into a much smaller *aggregate*.

3. Transformation

   These aggregates are then further processed to create an image.

Using this very general pipeline, many interesting data visualizations can be
created in a performant and scalable way. Datashader contains tools for easily
creating these pipelines in a composable manner, using only a few lines of code.


## Installation

Datashader is available on most platforms using the `conda` package manager,
from the `bokeh` channel:

```
conda install -c bokeh datashader
```

Alternatively, you can manually install from the repository:

```
git clone https://github.com/bokeh/datashader.git
cd datashader
conda install -c bokeh --file requirements.txt
python setup.py install
```

Datashader is not currently provided on pip/PyPI, to avoid broken or
low-performance installations that come from not keeping track of
C/C++binary dependencies such as LLVM (required by Numba).

One way to easily install `datashader` and related GIS and visualization tools is to install the conda environment from the `examples` directory of a local datashader repository clone:

```
cd examples
conda env create
source activate ds
```

Note on Windows to replace `source activate ds` with `activate ds`.

## Examples

There are lots of examples available in the `examples` directory, most of
which are viewable as notebooks on [Anaconda Cloud](https://anaconda.org/jbednar/notebooks).

## Learning more

Additional resources are linked from the
[datashader documentation](http://datashader.readthedocs.org), including
API documentation and papers and talks about the approach.

## Screenshots

![USA census](docs/images/usa_census.jpg)

![NYC races](docs/images/nyc_races.jpg)

![NYC taxi](docs/images/nyc_pickups_vs_dropoffs.jpg)
