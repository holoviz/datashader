# Datashader

[![Build Status](https://travis-ci.org/bokeh/datashader.svg)](https://travis-ci.org/bokeh/datashader)

The Datashader project is focused on building better ways to visualize
very large datasets, using intelligent server-side downsampling,
automatic computations performed on the data as it progresses through
the visualization pipeline, and other related techniques.  The project
is under active development, and all the code and documentation is
subject to frequent changes.

## Getting Started

```
# grab the master branch of the datashader repo
git clone https://github.com/bokeh/datashader.git

# Create a new conda environment
conda create -n datashader python=2.7
source activate datashader

# Install required packages
conda install pillow numba dynd-python pandas datashape

# Install the datashader library
cd datashader
python setup.py develop

# Install Bokeh for running examples
conda install -c https://conda.anaconda.org/bokeh/channel/dev bokeh
```

**Run the examples**

Currently requires taxi.castra/, which must be obtained separately, to be
placed into examples/data/.

```
# Start the server
cd examples
python dashboard.py
```

and then point your browser to the appropriate URL, which is localhost:5000 by default.
