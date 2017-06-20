# Datashader Examples

The best way to understand how Datashader works is to try out our
extensive set of examples. Static versions of most of them are
provided on [Anaconda Cloud](https://anaconda.org/jbednar/notebooks),
but for the full experience with dynamic updating you will need to
install them on a live server. To get started, first go to your home
directory and download the current list of everything needed for the
examples:

- Download the [conda ds environment file](https://raw.githubusercontent.com/bokeh/datashader/master/examples/environment.yml) and save it as `environment.yml`.

Then run the following commands in your terminal (command) prompt, from wherever you saved `environment.yml`:

```bash
1. conda env create --file environment.yml
2. source activate ds
3. python -c "from datashader import examples ; examples('datashader-examples')"
4. cd datashader-examples
5. python download_sample_data.py
```

Step 1 will read `environment.yml`, create a new Conda environment
named `ds`, and install of the libraries needed into that environment
(including datashader itself). It will use Python 3.6 by default, but
you can edit that file to specify a different Python version if you
prefer (which may require changing some of the dependencies in some
cases).

Step 2 will activate the `ds` environment, using it for all subsequent
commands. You will need to re-run step 2 after closing your terminal
or rebooting your machine, if you want to use anything in the `ds`
environment.  On Windows, you need to replace `source activate ds`
with `activate ds`.

Step 3 will copy the datashader examples from wherever Conda placed
them into a subdirectory `datashader-examples`.

Steps 4-5 will download the sample data required for the examples. The
total download size is currently about 3GB to transfer, requiring
about 9GB on disk when unpacked, which can take some time depending on
the speed of your connection.  The files involved are specified in the
text file `datasets.yml` in the `datashader-examples` directory, and
you are welcome to edit that file or to download the individual files
specified therein manually if you prefer, as long as you put them into
a subdirectory `data/` so the examples can find them.  Once these
steps have completed, you will be ready to run any of the examples
listed below.


## Notebooks

Most of the examples are in the form of runnable Jupyter notebooks. Copies of
these with all the images and output included are hosted at [Anaconda
Cloud](https://anaconda.org/jbednar/notebooks). To run these notebooks on your
own system, start up a Jupyter notebook server:

```
jupyter notebook --NotebookApp.iopub_data_rate_limit=100000000
```
(The data_rate setting here is required with Jupyter 5.0, but can be omitted for earlier or later versions).

If you want the generated notebooks to work without an internet connection or
with an unreliable connection (e.g. if you see `Loading BokehJS ...` but never
`BokekJS sucessfully loaded`), then restart the Jupyter notebook server using:

```
BOKEH_RESOURCES=inline jupyter notebook --NotebookApp.iopub_data_rate_limit=100000000
```

**[plotting_pitfalls](https://anaconda.org/jbednar/plotting_pitfalls/notebook)**

Motivation for the ideas behind datashader. Shows perceptual problems
that plotting in a conventional way can lead to. Re-running it locally
is usually not required, since the filled out version at the link
above has essentially the full data involved.

**[pipeline](https://anaconda.org/jbednar/pipeline/notebook)**

Step-by-step documentation for each of the stages in the datashader
pipeline, giving an overview of how to configure and use each
component provided.  Most useful when you have looked at the other
example dashboards and the notebooks below, and are ready to start
working with your own data.

**[nyc_taxi](https://anaconda.org/jbednar/nyc_taxi/notebook)**

Making geographical plots, with and without datashader, using trip data originally from
the [NYC Taxi dataset](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml)
but preprocessed using `taxi_preprocessing_example.py` for convenience.

**[census](https://anaconda.org/jbednar/census/notebook)**

Plotting the [2010 US Census data](http://www.coopercenter.org/demographics/Racial-Dot-Map), 
both to show population density and to show racial categories.

There is also a
[version showing how to visualize this data very simply using HoloViews](https://anaconda.org/jbednar/census-hv),
and a more complex one with additional dependencies that lets you
[compare congressional districts with racial categories]
(https://anaconda.org/jbednar/census-hv-dask).

**[holoviews_datashader](https://anaconda.org/jbednar/holoviews_datashader/notebook)**

How to use the separate [HoloViews](http://holoviews.org) package
to lay out and overlay datashader and non-datashader plots flexibly, 
making it simple to add dynamic datashader-based plots as needed.

**[nyc_taxi-nongeo](https://anaconda.org/jbednar/nyc_taxi-nongeo/notebook)**

Scatterplots for non-geographic variables in the taxi dataset.

**[tseries](https://anaconda.org/jbednar/tseries/notebook)**

Plotting large or multiple plots of time series (curve) data.

**[trajectory](https://anaconda.org/jbednar/trajectory/notebook)** and 
**[opensky](https://anaconda.org/jbednar/opensky/notebook)**

Plotting a 2D trajectory, either for a single long 
([random walk](https://anaconda.org/jbednar/trajectory/notebook)) or a
[large database of flight paths](https://anaconda.org/jbednar/opensky/notebook).

**[edge_bundling](https://anaconda.org/jbednar/edge_bundling/notebook)**

Plotting graph/network datasets, with or without bundling the edges
together to show structure.

**[2.7-billion-point OSM](https://anaconda.org/jbednar/osm/notebook)** and
**[1-billion-point OSM](https://anaconda.org/jbednar/osm-1billion)**.

Datashader supports [dask](http://dask.pydata.org) dataframes that
make it simple to work with out-of-core datasets (too large for the
physical memory on the machine) and distributed processing (across
cores or nodes). These examples show how to work with the 2.7 billion
GPS coordinates made available by
[Open Street Map](https://blog.openstreetmap.org/2012/04/01/bulk-gps-point-data/),
or a 1-billion-point subset of them that fits into memory on a 16GB
machine. 

**[Amazon.com center distance](https://anaconda.org/defusco/amz_centers/notebook)**

Cities in the USA colored by their distance to the nearest Amazon.com 
distribution center.

**[landsat](https://anaconda.org/jbednar/landsat/notebook)**,
**[race_elevation](https://anaconda.org/jbednar/race_elevation/notebook)**,
**[lidar](https://anaconda.org/jbednar/lidar/notebook)**, and
**[solar](https://anaconda.org/jbednar/solar/notebook)**

Various work-in-progress notebooks about using satellite, LIDAR, and
other weather/climate data with Datashader.


## Dashboard

An example interactive dashboard using
[bokeh server](http://bokeh.pydata.org/en/latest/docs/user_guide/server.html)
integrated with a datashading pipeline.

To start, launch it with one of the supported datasets specified:

```
python dashboard/dashboard.py -c dashboard/nyc_taxi.yml
python dashboard/dashboard.py -c dashboard/census.yml
python dashboard/dashboard.py -c dashboard/opensky.yml
python dashboard/dashboard.py -c dashboard/osm.yml
```

The '.yml' configuration file sets up the dashboard to use one of the
datasets downloaded above. You can write similar configuration files
for working with other datasets of your own, while adding features to
`dashboard.py` itself if needed to support them.

For most of these datasets, if you have less than 16GB of RAM on your
machine, you will want to add the "-o" option before "-c" to tell it
to work out of core instead of loading all data into memory.  However,
doing so will make interactive use substantially slower than if
sufficient memory were available.

To launch multiple dashboards at once, you'll need to add "-p 5001"
(etc.) to select a unique port number for the web page to use for
communicating with the Bokeh server.  Otherwise, be sure to kill the
server process before launching another instance.
