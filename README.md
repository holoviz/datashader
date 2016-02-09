Datashader
----------

[![Build Status](https://travis-ci.org/bokeh/datashader.svg)](https://travis-ci.org/bokeh/datashader)


Datashader is a graphics pipeline system for creating meaningful
representations of large amounts of data. It breaks the creation of images into
2 steps:

1. Aggregation

   Data is binned per-pixel by various user specified methods into aggregates.

2. Transformation

   These aggregates are then further processed to create an image.

Using this very general pipeline, many interesting data visualizations can be
created in a performant and scalable way. Datashader contains tools for easily
creating these pipelines in few lines of code.

The project is under active development, and all the code and documentation is
subject to frequent changes.

## Installation

```
conda install -c bokeh datashader
```

## Examples

Several example can be found in the `examples` directory.

## Related work

The core concepts of datashader are based off the concept of Abstract
Rendering:

- Abstract Rendering: [Out-of-core Rendering for Information
  Visualization](http://www.crest.iu.edu/publications/prints/2014/Cottam2014OutOfCore.pdf)
  (SPIE Conference on Visualization and Data Analysis 2014)
