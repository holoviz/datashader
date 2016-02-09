# Datashader Examples

The examples rely on some sample data. To download the data, run the
`download_sample_data.py` script. This may take up to 20 minutes, even on a
good network connection. The dataset is roughly 1.5 GB on disk.

```
python download_sample_data.py
```

The examples also require bokeh to be installed. Bokeh is available through
either conda or pip.

```
conda install bokeh

or

pip install bokeh
```

## Examples

### Dashboard

An example interactive dashboard using [bokeh
server](http://bokeh.pydata.org/en/latest/docs/user_guide/server.html)
integrated with a datashading pipeline. To start, run:

```
python dashboard/dashboard.py --config dashboard/nyc_taxi.yml
```

### Notebooks

Most of the examples are in the form of runnable Jupyter notebooks. Copies of
these with all the images and output included are hosted at [Anaconda
Cloud](https://anaconda.org/jbednar/notebooks). To run these notebooks on your
own system, install and startup a Jupyter notebook server:

```
conda install jupyter

or

pip install jupyter
```

To start:

```
jupyter notebook
```

**[plotting_problems](https://anaconda.org/jbednar/plotting_problems/notebook)**

Motivation for the ideas behind datashader. Shows perceptual problems that
plotting in a conventional way can lead to. Note that this example also
requires the holoviews package: `conda install -c ioam holoviews`.

**[nyc_taxi](https://anaconda.org/jbednar/nyc_taxi/notebook)**

Making geographical plots, with and without datashader, using trip data from
the [NYC Taxi dataset](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml).

**[nyc_taxi-nongeo](https://anaconda.org/jbednar/nyc_taxi-nongeo/notebook)**

A simple scatter plot on the taxi dataset.

**[osm](https://anaconda.org/jbednar/osm/notebook)**

Plotting the 2.7 billion gps coordinates made available by [open street
map](https://blog.openstreetmap.org/2012/04/01/bulk-gps-point-data/). This
dataset isn't provided by the download script, and is only included to
demonstrate working with a large dataset. The run notebook can be viewed using
the `anaconda.org` link provided above.
