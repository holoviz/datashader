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

A few example notebooks are also included. To run, install and startup a
`jupyter` notebook server:

```
conda install jupyter

or

pip install jupyter
```

To start:

```
jupyter notebook
```

**plotting_problems**

Motivation for the ideas behind datashader. Shows perceptual problems that
plotting in a conventional way can lead to.

**nyc_taxi**

Making geographical plots, with and without datashader, using trip data from
the [NYC Taxi dataset](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml).

**nyc_taxi-nongeo**

A simple scatter plot on the taxi dataset.

**osm**

Plotting the 2.7 billion gps coordinates made available by [open street
map](https://blog.openstreetmap.org/2012/04/01/bulk-gps-point-data/).
