# Datashader Examples

A large set of example notebooks and applications is provided with
Datashader, illustrating how to use it with a variety of external
libraries and datasets.

Once you follow the [steps for installing datashader](https://github.com/bokeh/datashader/blob/master/README.md#installation) ,
you can run the following commands from your terminal (command) prompt
to get a local copy of the examples, libraries, and datasets:

```bash
1. cd ~
2. python -c "from datashader import examples ; examples('datashader-examples')"
3. cd datashader-examples
4. conda env create --file environment.yml
5. source activate ds
6. python download_sample_data.py
```

(On Windows, replace `source activate ds` with `activate ds`.)

Steps 1-3 will copy the examples from wherever they ended up into a
subdirectory `datashader-examples` in your home directory.

Step 4 will read the file `environment.yml` included with the
examples, create a new Conda environment named `ds`, and install of
the dependencies listed in it into that environment. It will use
Python 3.6 by default, but you can edit that file to specify a
different Python version if you prefer (which may require changing
some of the dependencies in some cases).

Step 5 will activate the `ds` environment, using it for all subsequent
commands. You will need to re-run step 5 after closing your terminal
or rebooting your machine, if you want to use anything in the `ds`
environment.

Step 6 will download the sample data required for the examples. The total
download size is currently about 2.5GB to transfer, requiring about
7.5GB on disk when unpacked, which can take some time depending on the
speed of your connection.  The files involved are specified in the
text file `datasets.yml` in the examples directory, and you are welcome
to edit that file or to download the individual files specified therein
manually if you prefer, putting them into a subdirectory `data/`.

The "Census" example data is the largest file and should be the last
thing to be downloaded, so you should be able to start running all of
the other examples while that one completes.

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
`dashboard.py` itself if needed.

If you have less than 16GB of RAM on your machine, you will want to
add the "-o" option before "-c" to tell it to work out of core instead
of loading all data into memory, though doing so will make interactive
use substantially slower than if sufficient memory were available.

To launch multiple dashboards at once, you'll need to add "-p 5001"
(etc.) to select a unique port number for the web page to use for
communicating with the Bokeh server.  Otherwise, be sure to kill the
server process before launching another instance.
