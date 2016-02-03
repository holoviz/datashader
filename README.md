Datashader
----------

[![Build Status](https://travis-ci.org/bokeh/datashader.svg)](https://travis-ci.org/bokeh/datashader)

The Datashader project is focused on building better ways to visualize
very large datasets, using intelligent server-side downsampling,
automatic computations performed on the data as it progresses through
the visualization pipeline, and other related techniques.  The project
is under active development, and all the code and documentation is
subject to frequent changes.

## Installation

```
# Create a new conda environment, if desired
conda create -n datashader python=2.7
source activate datashader

# Install required packages, including latest fixes required
conda install numpy pandas pytest toolz numba datashape odo dask pillow
conda install -c dynd dynd-python
pip install --upgrade --no-deps git+https://github.com/Blaze/odo
pip install --upgrade --no-deps git+https://github.com/Blaze/datashape

# Install Bokeh for running examples
conda install -c https://conda.anaconda.org/bokeh/channel/dev bokeh

# Install the datashader library
git clone https://github.com/bokeh/datashader.git
cd datashader
python setup.py develop
```

**Running the examples**

```
cd examples
```

Download the sample data. This may take 20 minutes on a good connection, and more otherwise:
```
python download_sample_data.py
```

Dashboard example:
```
cd dashboard
python dashboard.py --config nyc_taxi.yml
cd ..
```

(and then point your browser to the appropriate URL, which is localhost:5000 by default.)

Jupyter notebook example:
```
jupyter notebook
```
(and then select `nyc_taxi.ipynb` from within the Jupyter notebook, and select `Cell/Run all` to create interactive plots.)
