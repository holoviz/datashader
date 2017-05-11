Performance
===========

Datashader is designed to make it simple to work with even very large
datasets.  To get good performance, it is essential that each step in
the overall processing pipeline be set up appropriately.  Below we
share some of our suggestions based on our own `Benchmarking`_ and
optimization experience, which should help you obtain suitable
performance in your own work.


File formats
------------

Based on our `testing with various file formats`_, we recommend
storing any large columnar datasets in the `Apache Parquet`_ format
when possible, using the `fastparquet`_ library with "Snappy" compression:

.. _`Benchmarking`: https://github.com/bokeh/datashader/issues/313
.. _`testing with various file formats`: https://github.com/bokeh/datashader/issues/129
.. _`Apache Parquet`: https://parquet.apache.org/
.. _`fastparquet`: https://github.com/dask/fastparquet

.. code-block:: python

    >>> import dask.dataframe as dd
    >>> dd.to_parquet(filename, df, compression="SNAPPY")

If your data includes categorical values that take on a limited, fixed
number of possible values (e.g. "Male", "Female"), With parquet,
categorical columns use a more memory-efficient data representation
and are optimized for common operations such as sorting and finding
uniques. Before saving, just convert the column as follows:

.. code-block:: python

    >>> df[colname] = df[colname].astype('category')

By default, numerical datasets typically use 64-bit floats, but many
applications do not require 64-bit precision when aggregating over a
very large number of datapoints to show a distribution.  Using 32-bit
floats reduces storage and memory requirements in half, and also
typically greatly speeds up computations because only half as much
data needs to be accessed in memory.  If applicable to your particular
situation, just convert the data type before generating the file:

.. code-block:: python

    >>> df[colname] = df[colname].astype(numpy.float32)


Single machine
--------------

Datashader supports both Pandas and Dask dataframes, but Dask
dataframes typically give higher performance even on a single machine,
because it makes good use of all available cores, and it also supports
out-of-core operation for datasets larger than memory.

Dasks works on chunks of the data at any one time, called partitions.
With dask on a single machine, a rule of thumb for the number of
partitions to use is ``multiprocessing.cpu_count()``, which allows
Dask to use one thread per core for parallelizing computations.

When the entire dataset fits into memory at once, you can persist the data
as a Dask dataframe prior to passing it into datashader, to ensure that 
data only needs to be loaded once:

.. code-block:: python

    >>> from dask import dataframe as dd
    >>> import multiprocessing as mp
    >>> dask_df = dd.from_pandas(df, npartitions=mp.cpu_count())
    >>> dask_df.persist()
    ...
    >>> cvs = datashader.Canvas(...)
    >>> agg = cvs.points(dask_df, ...)

When the entire dataset doesn't fit into memory at once, you should not
use `persist`.  In our tests of this scenario, dask's distributed scheduler gave 
better performance than the default scheduler, even on single machines:

.. code-block:: python

    >>> from dask import distributed
    >>> import multiprocessing as mp
    >>> cluster = distributed.LocalCluster(n_workers=mp.cpu_count(), threads_per_worker=1)
    >>> dask_client = distributed.Client(cluster)
    >>> dask_df = dd.from_pandas(df, npartitions=mp.cpu_count()) # Note no "persist"
    ...
    >>> cvs = datashader.Canvas(...)
    >>> agg = cvs.points(dask_df, ...)


Multiple machines
-----------------

To use multiple nodes (different machines) at once, you can use a Dask
dataframe with the distributed scheduler as shown
above. ``client.persist(dask_df)`` may help in certain cases, but if
you are doing profiling of the aggregation step, be sure to include
``distributed.wait()`` to block until the data is read into RAM on
each worker.
