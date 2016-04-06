# Datashader Examples

Many of the examples rely on the NYC Taxi dataset, which can be
downloaded by running the `download_sample_data.py` script. This may
take up to 20 minutes, even on a good network connection. The dataset
is roughly 1.5 GB on disk.

```
python download_sample_data.py
```

The examples also require bokeh to be installed. Bokeh is available through
either conda or pip.

```
conda install bokeh
```
or
```
pip install bokeh
```

Other dependencies for each example are listed below.


## Dashboard

An example interactive dashboard using
[bokeh server](http://bokeh.pydata.org/en/latest/docs/user_guide/server.html)
integrated with a datashading pipeline.  Requires webargs and (on Python2)
fastcache:

```
pip install webargs
conda install fastcache
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
pip install castra
```

You'll also need to download the 1.3GB data file
[census.castra.tar.gz](http://s3.amazonaws.com/bokeh_data/census.castra.tar.gz)
and unzip it into `examples/data/` (5GB unzipped), then run the
dashboard:

```
python dashboard/dashboard.py -c dashboard/census.yml
```

If you have other dashboards running, you'll need to add "-p 5001" (etc.) to select
a unique port number for the web page to use for communicating with the Bokeh server.

## Notebooks

Most of the examples are in the form of runnable Jupyter notebooks. Copies of
these with all the images and output included are hosted at [Anaconda
Cloud](https://anaconda.org/jbednar/notebooks). To run these notebooks on your
own system, install and start up a Jupyter notebook server:

```
conda install jupyter
```
or
```
pip install jupyter
```

To start:

```
jupyter notebook
```

**[plotting_pitfalls](https://anaconda.org/jbednar/plotting_pitfalls/notebook)**

Motivation for the ideas behind datashader. Shows perceptual problems that
plotting in a conventional way can lead to. Requires the HoloViews package:

```
conda install -c ioam holoviews
```

**[nyc_taxi](https://anaconda.org/jbednar/nyc_taxi/notebook)**

Making geographical plots, with and without datashader, using trip data from
the [NYC Taxi dataset](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml).

**[census](https://anaconda.org/jbednar/census/notebook)**

Plotting the 2010 US Census data, both to show population density and to show racial categories.
Requires the library and data files mentioned above.

**[nyc_taxi-nongeo](https://anaconda.org/jbednar/nyc_taxi-nongeo/notebook)**

Scatterplots for non-geographic variables in the taxi dataset.

**[tseries](https://anaconda.org/jbednar/tseries/notebook)**

Plotting large or multiple plots of time series (curve) data.

**[trajectory](https://anaconda.org/jbednar/trajectory/notebook)**

Plotting a 2D trajectory.

**[osm](https://anaconda.org/jbednar/osm/notebook)**

Plotting the 2.7 billion gps coordinates made available by [open street
map](https://blog.openstreetmap.org/2012/04/01/bulk-gps-point-data/). This
dataset isn't provided by the download script, and the notebook is only included to
demonstrate working with a large dataset. The run notebook can be viewed at
[anaconda.org](https://anaconda.org/jbednar/osm/notebook).
