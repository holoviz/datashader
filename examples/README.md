# Datashader Examples

The best way to understand how Datashader works is to try out our
extensive set of examples. [Datashader.org](https://datashader.org)
includes static versions of the
[getting started guide](https://datashader.org/getting_started),
[user manual](https://datashader.org/user_guide), and
[topic examples](https://datashader.org/topics), but for the full
experience with dynamic updating you will need to install them on a
live server.

These instructions assume you are using
[conda](https://conda.io/docs/install/quick.html), but they can be
adapted as needed to use [pip](https://pip.pypa.io/en/stable/installing/)
and [virtualenv](https://virtualenv.pypa.io) if desired.

To get started, run the following commands in your terminal (command) prompt:

```bash
git clone https://github.com/holoviz/datashader.git
cd datashader/examples
conda env create --file environment.yml
conda activate ds
```

The first two commands download the Datashader repository and go to
this `examples` folder.

`conda env create` reads `environment.yml`, creates a new Conda environment
named `ds`, and installs all of the libraries needed into that environment
(including datashader itself).

`conda activate ds` activates the `ds` environment, using it for all subsequent
commands. You will need to re-run it after closing your terminal or
rebooting your machine, if you want to use anything in the `ds` environment.

The notebooks create or download the data they need when you run them,
so once these steps have completed, you will be ready to run any of the
examples listed on [datashader.org](https://datashader.org).


## Notebooks

Most of the examples are in the form of runnable Jupyter
notebooks. Once you have obtained the notebooks, you can run them
on your own system by starting Jupyter in this `examples` folder:

```
jupyter notebook
```

If you want the generated notebooks to work without an internet connection or
with an unreliable connection (e.g. if you see `Loading BokehJS ...` but never
`BokehJS successfully loaded`), then restart the Jupyter notebook server using:

```
BOKEH_RESOURCES=inline jupyter notebook --NotebookApp.iopub_data_rate_limit=100000000
```

See dashboard.ipynb in this directory for a Datashder dashboard for viewing data.
