#!/usr/bin/env python3

"""
Simple test of read and write times for columnar data formats:
  python filetimes.py <filepath> [pandas|dask [hdf5base [xcolumn [ycolumn] [categories...]]]]

Test files may be generated starting from any file format supported by Pandas:
  python -c "import filetimes ; filetimes.base='<hdf5base>' ; filetimes.categories=['<cat1>','<cat2>']; filetimes.timed_write('<file>')"
"""

import time
global_start = time.time()

import io, os, os.path, sys, shutil, glob, argparse
import pandas as pd
import dask.dataframe as dd
import numpy as np
import datashader as ds
import bcolz
import feather
import dask
import fastparquet as fp

from datashader.utils import export_image
from datashader import transfer_functions as tf
from castra import Castra
from collections import OrderedDict as odict

#from multiprocessing.pool import ThreadPool
#dask.set_options(pool=ThreadPool(3)) # select a pecific number of threads

DEBUG = False

class Parameters(object):
    base,x,y='data','x','y'
    dftype='pandas'
    categories=[]
    chunksize=76668751
    parq_opts=dict(file_scheme='hive', has_nulls=0, write_index=False)
    cat_width=1 # Size of fixed-width string for representing categories
    columns=None
    cachesize=9e9

p=Parameters()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('filepath')
    parser.add_argument('dftype')
    parser.add_argument('base')
    parser.add_argument('x')
    parser.add_argument('y')
    parser.add_argument('categories', nargs='+')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    filepath = args.filepath
    basename, extension = os.path.splitext(filepath)
    p.dftype      = args.dftype
    p.base        = args.base
    p.x           = args.x
    p.y           = args.y
    p.categories  = args.categories
    DEBUG = args.debug


from dask.cache import Cache
Cache(p.cachesize).register()


filetypes_storing_categories = {'parq','castra'}


class Kwargs(dict):
    """Used to distinguish between dictionary argument values, and
    keyword-arguments.
    """
    pass

def benchmark(fn, args):
    """Benchmark when "fn" function gets called on "args" tuple.
    "args" may have a Kwargs instance at the end.
    """
    posargs = list(args)
    kwargs = {}
    # Remove Kwargs instance at end of posargs list, if one exists
    if posargs and isinstance(posargs[-1], Kwargs):
        lastarg = posargs.pop()
        kwargs.update(lastarg)

    if DEBUG:
        printable_posargs = ', '.join([str(posarg.head()) if hasattr(posarg, 'head') else str(posarg) for posarg in posargs])
        printable_kwargs = ', '.join(['{}={}'.format(k, v) for k,v in kwargs.items()])
        print('{}({}{})'.format(fn.__name__, printable_posargs, ', '+printable_kwargs if printable_kwargs else '', flush=True))

    # Benchmark fn when run on posargs and kwargs
    start = time.time()
    res = fn(*posargs, **kwargs)
    end = time.time()

    return end-start, res
    


read = odict([(f,odict()) for f in ["parq","bcolz","feather","castra","h5","csv"]])
               
read["csv"]     ["dask"]   = lambda filepath,p:  benchmark(dd.read_csv, (filepath,)) # Kwargs(usecols=p.columns)))
read["h5"]      ["dask"]   = lambda filepath,p:  benchmark(dd.read_hdf, (filepath, p.base, Kwargs(chunksize=p.chunksize, columns=p.columns)))
#read["castra"]  ["dask"]   = lambda filepath,p:  benchmark(dd.from_castra, (filepath,))
read["bcolz"]   ["dask"]   = lambda filepath,p:  benchmark(dd.from_bcolz, (filepath, Kwargs(chunksize=1000000)))
read["parq"]    ["dask"]   = lambda filepath,p:  benchmark(dd.read_parquet, (filepath, Kwargs(index=False, categories=p.categories, columns=p.columns)))

read["csv"]     ["pandas"] = lambda filepath,p:  benchmark(pd.read_csv, (filepath,)) # Kwargs(usecols=p.columns)))
read["h5"]      ["pandas"] = lambda filepath,p:  benchmark(pd.read_hdf, (filepath, p.base, Kwargs(columns=p.columns)))
read["feather"] ["pandas"] = lambda filepath,p:  benchmark(feather.read_dataframe, (filepath,))
def read_parq_pandas(__filepath):
    return fp.ParquetFile(__filepath).to_pandas()
read["parq"]    ["pandas"] = lambda filepath,p:  benchmark(read_parq_pandas, (filepath,))


write = odict([(f,odict()) for f in ["parq","snappy.parq","gz.parq","bcolz","feather","castra","h5","csv"]])

write["csv"]          ["dask"]   = lambda df,filepath,p:  benchmark(df.to_csv, (filepath.replace(".csv","*.csv"),))
write["h5"]           ["dask"]   = lambda df,filepath,p:  benchmark(df.to_hdf, (filepath, p.base))
#write["castra"]       ["dask"]   = lambda df,filepath,p:  benchmark(df.to_castra, (filepath, Kwargs(categories=p.categories)))
write["parq"]         ["dask"]   = lambda df,filepath,p:  benchmark(dd.to_parquet, (filepath, df)) # **p.parq_opts
write["snappy.parq"]  ["dask"]   = lambda df,filepath,p:  benchmark(dd.to_parquet, (filepath, df, Kwargs(compression='SNAPPY'))) ## **p.parq_opts
write["gz.parq"]      ["dask"]   = lambda df,filepath,p:  benchmark(dd.to_parquet, (filepath, df, Kwargs(compression='GZIP')))

write["csv"]          ["pandas"] = lambda df,filepath,p:  benchmark(df.to_csv, (filepath,))
write["h5"]           ["pandas"] = lambda df,filepath,p:  benchmark(df.to_hdf, (filepath, Kwargs(key=p.base, format='table')))

def write_castra_pandas(__filepath, __df, __cats):
    return Castra(__filepath, template=__df, categories=__cats).extend(__df)
write["castra"]       ["pandas"] = lambda df,filepath,p:  benchmark(write_castra_pandas, (filepath, df, p.categories))
write["bcolz"]        ["pandas"] = lambda df,filepath,p:  benchmark(bcolz.ctable.fromdataframe, (df, Kwargs(rootdir=filepath)))
write["feather"]      ["pandas"] = lambda df,filepath,p:  benchmark(feather.write_dataframe, (df, filepath))
write["parq"]         ["pandas"] = lambda df,filepath,p:  benchmark(fp.write, (filepath, df, p.parq_opts))
write["snappy.parq"]  ["pandas"] = lambda df,filepath,p:  benchmark(fp.write, (filepath, df, Kwargs(compression='SNAPPY', **p.parq_opts)))
#write["gz.parq"]      ["pandas"] = lambda df,filepath,p:  benchmark(fp.write, (filepath, df, Kwargs(fixed_text={c:p.cat_width for c in p.categories}, compression='GZIP', **p.parq_opts)))


def timed_write(filepath,dftype,output_directory="times"):
    """Accepts any file with a dataframe readable by the given dataframe type, and writes it out as a variety of file types"""
    df,duration=timed_read(filepath,dftype)

    for ext in write.keys():
        directory,filename = os.path.split(filepath)
        basename, extension = os.path.splitext(filename)
        fname = output_directory+os.path.sep+basename+"."+ext
        if os.path.exists(fname):
            print("{:28} (keeping existing)".format(fname))
        else:
            filetype=ext.split(".")[-1]
            if not filetype in filetypes_storing_categories:
                for c in p.categories:
                    if filetype == 'parq' and df[c].dtype == 'object':
                        df[c]=df[c].str.encode('utf8')
                    else:
                        df[c]=df[c].astype(str)

            code = write[ext].get(dftype,None)

            if code is None:
                print("{:28} {:7} Not supported".format(fname,dftype))
            else:
                duration, res = code(df,fname,p)
                print("{:28} {:7} {:05.2f}".format(fname,dftype,duration))

            if not filetype in filetypes_storing_categories:
                for c in p.categories:
                    df[c]=df[c].astype('category')

        
def timed_read(filepath,dftype):
    basename, extension = os.path.splitext(filepath)
    extension = extension[1:]
    filetype=extension.split(".")[-1]
    code = read[extension].get(dftype,None)

    if filetype=="csv" and dftype=="dask":
        filepath = filepath.replace(".csv","*.csv")
    
    if code is None or not glob.glob(filepath):
        return (None,None)
    
    p.columns=[p.x]+[p.y]+p.categories
    
    duration, df = code(filepath,p)

    if not filetype in filetypes_storing_categories:
        opts=odict()
        if dftype == 'pandas':
            opts['copy']=False
        for c in p.categories:
            df[c]=df[c].astype('category',**opts)
    
    if dftype=='dask':
        # Force loading
        # df = dd.from_pandas(df.compute(), npartitions=4)
        pass
    
    return df, duration


def timed_agg(df, filepath, plot_width=int(900), plot_height=int(900*7.0/12)):
    start = time.time()
    cvs = ds.Canvas(plot_width, plot_height)
    agg = cvs.points(df, p.x, p.y)
    end = time.time()
    img = export_image(tf.shade(agg),filepath,export_path=".")
    return img, end-start


def get_size(path):
    total = os.path.getsize(path) if os.path.isfile(path) else 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total


if __name__ == '__main__':
    df,loadtime = timed_read(filepath,p.dftype)

    if df is None:
        print("{:28} {:6}  Not supported".format(filepath, p.dftype))
        sys.exit(1)

    img,aggtime1 = timed_agg(df,filepath,5,5)
    img,aggtime2 = timed_agg(df,filepath)
    
    in_size  = get_size(filepath)
    out_size = get_size("{}.png".format(filepath))
    
    global_end = time.time()
    print("{:28} {:6}  Aggregate1:{:06.2f} ({:06.2f}+{:06.2f})  Aggregate2:{:06.2f}  In:{:011d}  Out:{:011d}  Total:{:06.2f}"\
          .format(filepath, p.dftype, loadtime+aggtime1, loadtime, aggtime1, aggtime2, in_size, out_size, global_end-global_start))
