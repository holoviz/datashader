Benchmarking
============

`Datashader` uses ASV (https://asv.readthedocs.io) for benchmarking.

Preparation
-----------

ASV runs benchmarks in isolated virtual environments that it creates. It identifies whether you are running in a `conda` or `virtualenv` environment so that it knows what type of environment to use. Before you run any benchmarks you need to install `asv` itself; if you are using `conda`:
```
conda install -c conda-forge asv==0.4.2
```

and if you are using `virtualenv`:
```
pip install asv==0.4.2 virtualenv
```

Running benchmarks
------------------

To run all benchmarks against the default `master` branch:
```
cd benchmarks
asv run
```

The first time this is run it will create a machine file to store information about your machine. Then a virtual environment will be created and each benchmark will be run multiple times to obtain a statistically valid benchmark time.

To list the benchmark timings stored for the `master` branch use:
```
asv show master
```

ASV ships with its own simple webserver to interactively display the results in a webbrowser. To use this:
```
asv publish
asv preview
```
and then open a web browser at the URL specified.

If you want to quickly run all benchmarks once only to check for errors, etc, use:
```
asv dev
```
instead of `asv run`.


Running cuDF and Dask-cuDF benchmarks
-------------------------------------

Benchmarks that use `pandas` and `dask` DataFrames are always run whereas those that use `cudf` and `dask-cudf` are only run if the required libraries are installed and appropriate GPU hardware is available. Because installing the required libraries is non-trivial it is recommended to run the benchmarks in your default `cudf`-enabled development environment rather than allow `asv` to create new environments specifically for the benchmarking.

Before running `cudf` and `dask-cudf` benchmarks you should first check that you can run the Datashader `pytest` test suite as debugging your environment is much easier using `pytest` than `asv`.

The `asv` command to run all benchmarks using your default development environment is:
```
asv run --python=same --launch-method spawn
```

The `--launch-method spawn` is recommended to avoid problems in accessing the GPU from subprocesses which is how `asv` runs individual isolated benchmarks.


Adding new benchmarks
---------------------

Add new benchmarks to existing or new classes in the `benchmarks/benchmarks` directory. Any class member function with a name that starts with `time` will be identified as a timing benchmark when `asv` is run.

Data that is required to run benchmarks is usually created in the `setup()` member function. This ensures that the time taken to setup the data is not included in the benchmark time. The `setup()` function is called once for each invocation of each benchmark, the data are not cached.

At the top of each benchmark class there are lists of parameter names and values. Each benchmark is repeated for each unique combination of these parameters.

If you only want to run a subset of benchmarks, use syntax like:
```
asv run -b ShadeCategorical
```
where the text after the `-b` flag is used as a regex to match benchmark file, class and function names.


Benchmarking code changes
-------------------------

You can compare the performance of code on different branches and in different commits. Usually if you want to determine how much faster a new algorithm is, the old code will be in the `master` branch and the new code will be in a new feature branch. Because ASV uses virtual environments and checks out the `datashader` source code into these virtual environments, your new code must be committed into the new feature branch locally.

To benchmark the latest commits on `master` and your new feature branch, edit `asv.conf.json` to change the line
```
"branches": ["master"],
```
into
```
"branches": ["master", "new_feature_branch"],
```
or similar.

Now when you `asv run` the benchmarks will be run against both branches in turn.

Then use
```
asv show
```
to list the commits that have been benchmarked, and
```
asv compare commit1 commit2
```
to give you a side-by-side comparison of the two commits.
