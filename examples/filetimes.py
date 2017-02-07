#!/usr/bin/env python3
"""Simple test of read and write times for columnar data formats:
  python filetimes.py <filepath> [pandas|dask [hdf5base [xcolumn [ycolumn] [categories...]]]]

Test files may be generated starting from any file format supported by Pandas:
  python -c "import filetimes ; filetimes.base='<hdf5base>' ; filetimes.categories=['<cat1>','<cat2>']; filetimes.timed_write('<file>')"
"""

import time
global_start = time.time()

import io, os, os.path, sys, shutil, glob
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
    if len(sys.argv)<=1:
        print(__doc__)
        sys.exit(1)

    filepath = sys.argv[1]
    basename, extension = os.path.splitext(filepath)

    if len(sys.argv)>2: p.dftype      = sys.argv[2]
    if len(sys.argv)>3: p.base        = sys.argv[3]
    if len(sys.argv)>4: p.x           = sys.argv[4]
    if len(sys.argv)>5: p.y           = sys.argv[5]
    if len(sys.argv)>6: p.categories  = sys.argv[6:]

from dask.cache import Cache
Cache(p.cachesize).register()


filetypes_storing_categories = {'parq','castra'}


read = odict([(f,odict()) for f in ["parq","bcolz","feather","castra","h5","csv"]])
               
read["csv"]     ["dask"]   = lambda filepath,p:  dd.read_csv(filepath, usecols=p.columns)
read["h5"]      ["dask"]   = lambda filepath,p:  dd.read_hdf(filepath, p.base, chunksize=p.chunksize, columns=p.columns)
read["castra"]  ["dask"]   = lambda filepath,p:  dd.from_castra(filepath)
read["bcolz"]   ["dask"]   = lambda filepath,p:  dd.from_bcolz(filepath, chunksize=1000000)
read["parq"]    ["dask"]   = lambda filepath,p:  dd.io.parquet.read_parquet(filepath,index=False, categories=p.categories, columns=p.columns)

read["csv"]     ["pandas"] = lambda filepath,p:  pd.read_csv(filepath, usecols=p.columns)
read["h5"]      ["pandas"] = lambda filepath,p:  pd.read_hdf(filepath, p.base, columns=p.columns)
read["feather"] ["pandas"] = lambda filepath,p:  feather.read_dataframe(filepath)
read["parq"]    ["pandas"] = lambda filepath,p:  fp.ParquetFile(filepath).to_pandas()


write = odict([(f,odict()) for f in ["parq","snappy.parq","gz.parq","bcolz","feather","castra","h5","csv"]])

write["csv"]          ["dask"]   = lambda df,filepath,p:  df.to_csv(filepath.replace(".csv","*.csv"))
write["h5"]           ["dask"]   = lambda df,filepath,p:  df.to_hdf(filepath, p.base)
write["castra"]       ["dask"]   = lambda df,filepath,p:  df.to_castra(filepath,categories=p.categories)
write["parq"]         ["dask"]   = lambda df,filepath,p:  dd.io.parquet.to_parquet(filepath, df) ## **p.parq_opts
write["snappy.parq"]  ["dask"]   = lambda df,filepath,p:  dd.io.parquet.to_parquet(filepath, df, compression='SNAPPY') ## **p.parq_opts
#write["gz.parq"]      ["dask"]   = lambda df,filepath,p:  dd.io.parquet.to_parquet(filepath, df, compression='GZIP')

write["csv"]          ["pandas"] = lambda df,filepath,p:  df.to_csv(filepath)
write["h5"]           ["pandas"] = lambda df,filepath,p:  df.to_hdf(filepath,key=p.base,format='table')
write["castra"]       ["pandas"] = lambda df,filepath,p:  Castra(filepath, template=df,categories=p.categories).extend(df)
write["bcolz"]        ["pandas"] = lambda df,filepath,p:  bcolz.ctable.fromdataframe(df, rootdir=filepath)
write["feather"]      ["pandas"] = lambda df,filepath,p:  feather.write_dataframe(df, filepath)
write["parq"]         ["pandas"] = lambda df,filepath,p:  fp.write(filepath, df, **p.parq_opts)
write["snappy.parq"]  ["pandas"] = lambda df,filepath,p:  fp.write(filepath, df, compression='SNAPPY', **p.parq_opts)
#write["gz.parq"]      ["pandas"] = lambda df,filepath,p:  fp.write(filepath, df, fixed_text={c:p.cat_width for c in p.categories}, compression='GZIP', **p.parq_opts)


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
                    if filetype=='parq' and dftype=='pandas':
                        df[c]=df[c].str.encode('utf8')
                    else:
                        df[c]=df[c].astype(str)

            code = write[ext].get(dftype,None)

            if code is None:
                print("{:28} {:7} Not supported".format(fname,dftype))
            else:
                start = time.time()
                code(df,fname,p)
                end = time.time()
                print("{:28} {:7} {:05.2f}".format(fname,dftype,end-start))

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
    
    start = time.time()
    p.columns=[p.x]+[p.y]+p.categories
    
    df = code(filepath,p)

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
    
    end = time.time()

    return df, end-start


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

