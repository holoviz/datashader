# Datashader Examples

The examples rely on some sample data. To download the data, run the
`download_sample_data.py` script. This may take up to 20 minutes, even on a
good network connection. The dataset is roughly 1.5 GB on disk.

```
python download_sample_data.py
```

Datashader is an independent library, but most of the examples require
bokeh to be installed. Bokeh is available through either conda or pip:

```
conda install bokeh

or

pip install bokeh
```

## Examples

### Notebooks

Most of the examples are in the form of runnable Jupyter notebooks. Copies of
these with all the images and output included are hosted at [Anaconda
Cloud](https://anaconda.org/jbednar/notebooks). To run these notebooks on your
own system, install and start up a Jupyter notebook server:

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



### Dashboard

An example interactive dashboard using 
[bokeh server](http://bokeh.pydata.org/en/latest/docs/user_guide/server.html)
integrated with a datashading pipeline.  Requires webargs:

```
pip install webargs
```

To start, run:

```
python dashboard/dashboard.py -c dashboard/nyc_taxi.yml
```

The 'nyc_taxi.yml' configuration file set up the dashboard to use the
NYC Taxi dataset downloaded above.  If you have less than 16GB of RAM
on your machine, you will want to add the "-o" option to tell it to work
out of core instead of loading all data into memory, though doing so will
make interactive use substantially slower than if sufficient memory were
available.

You can write similar configuration files for working with other
datasets of your own, while adding features to dashboard.py itself if
needed.  As an example, a configuration file for the [2010 US Census
racial data](http://www.coopercenter.org/demographics/Racial-Dot-Map)
is also provided.  To use the census data, you'll need to install dask
and castra:

```
conda install dask
conda install -c quanyuan castra
```

You'll also need to download the 1.3GB data file
[census.castra.zip]()
and unzip it into `examples/data/` (5GB unzipped), then run the
dashboard:

```
python dashboard/dashboard.py -c dashboard/census.yml
```

If you have other dashboards running, you'll need to add "-p 5001" (etc.) to select
a unique port number for the web page to use for communicating with the Bokeh server.
