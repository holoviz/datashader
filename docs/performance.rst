Performance
===========

General
-------

When interacting with data on the filesystem, store it in the
`Apache Parquet`_ format when possible. Snappy compression should be
used when writing out parquet files, and the data should rely on
categorical dtypes (when possible) before writing the parquet files,
as parquet supports categoricals in its binary format.

.. _`Apache Parquet`: https://parquet.apache.org/

.. code-block:: python

    >>> import dask.dataframe as dd
    >>> dd.to_parquet(filename, df, compression="SNAPPY")

In addition, use the categorical dtype for columns with data that takes
on a limited, fixed number of possible values. Categorical columns use a
more memory-efficient data representation and are optimized for common
operations such as sorting and finding uniques. Example of how to
convert a column to the categorical dtype:

.. code-block:: python

    >>> df[colname] = df[colname].astype('category')

There is also promise with improving datashader's performance even
further by using single-precision floats (``numpy.float32``) instead of
double-precision floats (``numpy.float64``). In past experiments this
cut down the time to load data off of disk (assuming the data was
written out in single-precision float) as well as datashader's
aggregation times. Care should be taken using this approach, as using
single-precision (in any software application, not just datashader)
leads to different numerical results than double-precision.

.. code-block:: python

    >>> df[colname] = df[colname].astype(numpy.float32)


Single machine
--------------

A rule-of-thumb for the number of partitions to use while converting
Pandas dataframes into Dask dataframes is ``multiprocessing.cpu_count()``.
This allows Dask to use one thread per core for parallelizing
computations.

When the entire dataset fits into memory at once, persist the dataframe
as a Dask dataframe prior to passing it into datashader. One example of
how to do this:

.. code-block:: python

    >>> from dask import dataframe as dd
    >>> import multiprocessing as mp
    >>> dask_df = dd.from_pandas(df, npartitions=mp.cpu_count())
    >>> dask_df.persist()
    ...
    >>> cvs = datashader.Canvas(...)
    >>> agg = cvs.points(dask_df, ...)

When the entire dataset doesn't fit into memory at once, use the
distributed scheduler without persisting. For example:

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

Use the distributed scheduler to farm computations out to remote
machines. ``client.persist(dask_df)`` may help in certain cases, but be
sure to include ``distributed.wait()`` to block until the data is read
into RAM on each worker.
