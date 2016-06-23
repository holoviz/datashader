# Datashader Examples

Some of the examples create their own synthetic datasets, but others
require real-world sample data that will need to be downloaded first.
The total download size is currently about 2.5GB to transfer,
requiring about 7.5GB on disk when unpacked, which can take some time
depending on the speed of your connection.  The files involved are
specified in the text file `datasets.yml` in this directory and can be
downloaded individually by hand if you prefer.  However, the easiest
approach is just to get them all at once, by running the
`datashader-download-data` command that is installed with datashader:

```bash
datashader-download-data
```
or else (e.g. if you have a git checkout) you can call the Python
script directly:

```bash
cd examples
python download_sample_data.py
```

The "Census" example data is the largest file and should be the last
thing to be downloaded, so you should be able to start running all of
the other examples while that one completes.

Most of the examples also require `bokeh` to be installed. 
Bokeh is available through either conda or pip:

```
conda install bokeh
```
or
```
pip install bokeh
```

Specific examples may have other dependencies as listed below.


## Dashboard

An example interactive dashboard using
[bokeh server](http://bokeh.pydata.org/en/latest/docs/user_guide/server.html)
integrated with a datashading pipeline.  Requires webargs and (on Python2)
fastcache:

```
pip install webargs
conda install fastcache
```

To start, launch it with one of the supported datasets specified:

```
python dashboard/dashboard.py -c dashboard/nyc_taxi.yml
python dashboard/dashboard.py -c dashboard/census.yml
```

The '.yml' configuration file sets up the dashboard to use one of the
datasets downloaded above. You can write similar configuration files
for working with other datasets of your own, while adding features to
`dashboard.py` itself if needed.

If you have less than 16GB of RAM on your machine, you will want to
add the "-o" option before "-c" to tell it to work out of core instead
of loading all data into memory, though doing so will make interactive
use substantially slower than if sufficient memory were available.

To launch multiple dashboards at once, you'll need to add "-p 5001"
(etc.) to select a unique port number for the web page to use for
communicating with the Bokeh server.

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

Motivation for the ideas behind datashader. Shows perceptual problems
that plotting in a conventional way can lead to. Re-running it locally
is usually not required, since the filled out version at the link
above has the full data. If you do wish to re-run it, you will need to
install the HoloViews package:

```
conda install -c ioam holoviews
```

**[pipeline](https://anaconda.org/jbednar/pipeline/notebook)**

Step-by-step documentation for each of the stages in the datashader
pipeline, giving an overview of how to configure and use each
component provided.  Most useful when you have looked at the other
example dashboards and the notebooks below, and are ready to start
working with your own data.

```
conda install -c ioam holoviews
```

**[nyc_taxi](https://anaconda.org/jbednar/nyc_taxi/notebook)**

Making geographical plots, with and without datashader, using trip data from
the [NYC Taxi dataset](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml).

**[census](https://anaconda.org/jbednar/census/notebook)**

Plotting the [2010 US Census data](http://www.coopercenter.org/demographics/Racial-Dot-Map), 
both to show population density and to show racial categories. 

**[nyc_taxi-nongeo](https://anaconda.org/jbednar/nyc_taxi-nongeo/notebook)**

Scatterplots for non-geographic variables in the taxi dataset.

**[tseries](https://anaconda.org/jbednar/tseries/notebook)**

Plotting large or multiple plots of time series (curve) data.

**[trajectory](https://anaconda.org/jbednar/trajectory/notebook)**

Plotting a 2D trajectory.

**[census](https://anaconda.org/jbednar/race_elevation/notebook)**

Combining raster data with scatterpoint data, using the 
census data on race along with gridded elevation data for Austin, TX.
Requires rasterio (`conda install rasterio krb5`).

**[osm](https://anaconda.org/jbednar/osm/notebook)**

Plotting the 2.7 billion gps coordinates made available by [open street
map](https://blog.openstreetmap.org/2012/04/01/bulk-gps-point-data/). This
dataset is not provided by the download script, and the notebook is only included to
demonstrate working with a large dataset. Requires `dask` for
out-of-core operation and `castra` for the fast file format.  The run
notebook can be viewed at
[anaconda.org](https://anaconda.org/jbednar/osm/notebook).
