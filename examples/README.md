# Datashader Examples

A variety of example notebooks and applications are maintained in the
examples/ subdirectory of Datashader's git repository, and these will be
installed somewhere on your local system when you install Datashader.
To get a copy of the examples in your own directory so that you can
run and edit them, you can run these commands in your terminal:

```bash
1. cd ~
2. python -c "from datashader import examples ; examples('datashader-examples')"
3. cd datashader-examples
4. python download_sample_data.py
```

Steps 1-3 steps will give you a copy of the notebooks and apps, and
step 4 will download various datasets used by them.  The total
download size is currently about 2.5GB to transfer, requiring about
7.5GB on disk when unpacked, which can take some time depending on the
speed of your connection.  The files involved are specified in the
text file `datasets.yml` that was copied to your directory in step 3,
and instead of step 4 you can download each file individually if you
prefer.

The "Census" example data is the largest file and should be the last
thing to be downloaded, so you should be able to start running all of
the other examples while that one completes.

Datashader itself is independent of other plotting libraries, but many of
the examples do use various plotting libraries, including Bokeh, 
HoloViews, and matplotlib.  To install these libraries and other
dependencies, you can run:

```
conda env create --file examples/environment.yml
source activate ds
```

(On Windows, replace `source activate ds` with `activate ds`.)


The dashboard example has additional dependencies as listed below.

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
above has the full data.

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

There is also a [version that lets you compare congressional districts with racial categories]
(https://anaconda.org/jbednar/census-hv-dask), which has its own installation
requirements because of overlaying shape files.

**[nyc_taxi-nongeo](https://anaconda.org/jbednar/nyc_taxi-nongeo/notebook)**

Scatterplots for non-geographic variables in the taxi dataset.

**[tseries](https://anaconda.org/jbednar/tseries/notebook)**

Plotting large or multiple plots of time series (curve) data.

**[trajectory](https://anaconda.org/jbednar/trajectory/notebook)** and 
**[opensky](https://anaconda.org/jbednar/opensky/notebook)**

Plotting a 2D trajectory, either for a single long 
([random walk](https://anaconda.org/jbednar/trajectory/notebook)) or a
[large database of flight paths](https://anaconda.org/jbednar/opensky/notebook).

**[landsat](https://anaconda.org/jbednar/landsat/notebook)** and
**[race_elevation](https://anaconda.org/jbednar/race_elevation/notebook)**

Combining raster data with scatterpoint data, using the 
census data on race along with gridded elevation data for Austin, TX.

**[osm](https://anaconda.org/jbednar/osm/notebook)**

Plotting the 2.7 billion GPS coordinates made available by [open street
map](https://blog.openstreetmap.org/2012/04/01/bulk-gps-point-data/). This
dataset is not provided by the download script, and the notebook is only
included to demonstrate working with a large dataset. The run notebook can be
viewed at [anaconda.org](https://anaconda.org/jbednar/osm). A 
[1-billion-point subset](https://anaconda.org/jbednar/osm-1billion) is also 
available for separate download.

**[Amazon.com center distance](https://anaconda.org/defusco/amz_centers/notebook)**

Cities in the USA colored by their distance to the nearest Amazon.com 
distribution center.


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
python dashboard/dashboard.py -c dashboard/opensky.yml
python dashboard/dashboard.py -c dashboard/osm.yml
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
communicating with the Bokeh server.  Otherwise, be sure to kill the
server process before launching another instance.
