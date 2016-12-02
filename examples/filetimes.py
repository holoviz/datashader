#!/usr/bin/env python3
"""Simple test of read and write times for columnar data formats:
  python filetimes.py <filepath> [pandas|dask [hdf5base [xcolumn [ycolumn] [categoricals...]]]]

Test files may be generated starting from any file format supported by Pandas:
  python -c "import filetimes ; filetimes.base='<hdf5base>' ; filetimes.categories=['<cat1>','<cat2>']; filetimes.timed_write('<file>')"
"""

import io, os, os.path, sys, time, shutil
import pandas as pd
import dask.dataframe as dd
import numpy as np
import datashader as ds
import bcolz
import feather
import fastparquet as fp

from datashader.utils import export_image
from datashader import transfer_functions as tf
from castra import Castra
from collections import OrderedDict as odict
from dask.cache import Cache

cachesize=9e9
base,x,y='data','x','y'
dftype='pandas'
categories=[]
chunksize=76668751
parq_opts=dict(file_scheme='hive', has_nulls=0, write_index=False)
cat_width=1 # Size of fixed-width string for representing categories

Cache(9e9).register()

filetypes_storing_categories = {'parq','castra'}


read = odict(csv=odict(), h5=odict(), castra=odict(), bcolz=odict(), feather=odict(), parq=odict())

read["csv"]     ["dask"]   = lambda filepath:  dd.read_csv(filepath)
read["h5"]      ["dask"]   = lambda filepath:  dd.read_hdf(filepath, base, chunksize=chunksize)
read["castra"]  ["dask"]   = lambda filepath:  dd.from_castra(filepath)
read["bcolz"]   ["dask"]   = lambda filepath:  dd.from_bcolz(filepath, chunksize=1000000)
read["parq"]    ["dask"]   = lambda filepath:  dd.io.parquet.read_parquet(filepath,index=False, categories=categories)

read["csv"]     ["pandas"] = lambda filepath:  pd.read_csv(filepath)
read["h5"]      ["pandas"] = lambda filepath:  pd.read_hdf(filepath, base)
read["feather"] ["pandas"] = lambda filepath:  feather.read_dataframe(filepath)
read["parq"]    ["pandas"] = lambda filepath:  fp.ParquetFile(filepath).to_pandas()


write = odict(csv=odict(), h5=odict(), castra=odict(), bcolz=odict(), feather=odict(), parq=odict())
write["snappy.parq"]=odict()
write["gz.parq"]=odict()

write["csv"]          ["pandas"] = lambda df,filepath:  df.to_csv(filepath)
write["h5"]           ["pandas"] = lambda df,filepath:  df.to_hdf(filepath,key=base,format='table')
write["castra"]       ["pandas"] = lambda df,filepath:  Castra(filepath, template=df,categories=categories).extend(df)
write["bcolz"]        ["pandas"] = lambda df,filepath:  bcolz.ctable.fromdataframe(df, rootdir=filepath)
write["feather"]      ["pandas"] = lambda df,filepath:  feather.write_dataframe(df, filepath)
write["parq"]         ["pandas"] = lambda df,filepath:  fp.write(filepath, df, file_scheme='hive')
write["parq"]         ["pandas"] = lambda df,filepath:  fp.write(filepath, df, fixed_text={c:cat_width for c in categories}, **parq_opts)
write["snappy.parq"]  ["pandas"] = lambda df,filepath:  fp.write(filepath, df, fixed_text={c:cat_width for c in categories}, compression='SNAPPY', **parq_opts)
write["gz.parq"]      ["pandas"] = lambda df,filepath:  fp.write(filepath, df, fixed_text={c:cat_width for c in categories}, compression='GZIP', **parq_opts)

write["h5"]           ["dask"]   = lambda df,filepath:  df.to_hdf(filepath, base)
write["parq"]         ["dask"]   = lambda df,filepath:  dd.io.parquet.to_parquet(filepath, df)
write["snappy.parq"]  ["dask"]   = lambda df,filepath:  dd.io.parquet.to_parquet(filepath, df, compression='SNAPPY')
write["gz.parq"]      ["dask"]   = lambda df,filepath:  dd.io.parquet.to_parquet(filepath, df, compression='GZIP')


def timed_write(filepath,output_directory="times",dftype=dftype):
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
                for c in categories:
                    if filetype=='parq' and dftype=='pandas':
                        df[c]=df[c].str.encode('utf8')
                    else:
                        df[c]=df[c].astype(str)

            code = write[ext].get(dftype,None)

            if code is None:
                print("{:28} {:7} Not supported".format(fname,dftype))
            else:
                start = time.time()
                code(df,fname)
                end = time.time()
                print("{:28} {:7} {:05.2f}".format(fname,dftype,end-start))

            if not filetype in filetypes_storing_categories:
                for c in categories:
                    df[c]=df[c].astype('category')

        
def timed_read(filepath,dftype):
    basename, extension = os.path.splitext(filepath)
    extension = extension[1:]
    filetype=extension.split(".")[-1]
    code = read[extension].get(dftype,None)
    if code is None or not os.path.exists(filepath):
        return (None,None)
    
    start = time.time()
    df = code(filepath)
    
    if not filetype in filetypes_storing_categories:
        opts={}
        if dftype == 'pandas':
            opts=dict(copy=False)
        for c in categories:
            df[c]=df[c].astype('category',**opts)
    
    end = time.time()

    return df, end-start


def timed_agg(df, filepath, plot_width=int(900), plot_height=int(900*7.0/12)):
    start = time.time()
    cvs = ds.Canvas(plot_width, plot_height)
    agg = cvs.points(df, x, y)
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
    if len(sys.argv)<=1:
        print(__doc__)
        sys.exit(1)

    filepath = sys.argv[1]
    basename, extension = os.path.splitext(filepath)

    if len(sys.argv)>2: dftype      = sys.argv[2]
    if len(sys.argv)>3: base        = sys.argv[3]
    if len(sys.argv)>4: x           = sys.argv[4]
    if len(sys.argv)>5: y           = sys.argv[5]
    if len(sys.argv)>6: categories  = sys.argv[6:]

    df,loadtime = timed_read(filepath,dftype)

    if df is None:
        print("{:28} {:6}  Not supported".format(filepath, dftype))
        sys.exit(1)

    img,aggtime1 = timed_agg(df,filepath,5,5)
    img,aggtime2 = timed_agg(df,filepath)
    
    in_size  = get_size(filepath)
    out_size = get_size("{}.png".format(filepath))

    print("{:28} {:6}  Total:{:06.2f}  Load:{:06.2f}  Aggregate1:{:06.2f}  Aggregate2:{:06.2f}  In:{:011d}  Out:{:011d}"\
          .format(filepath, dftype, loadtime+aggtime1, loadtime, aggtime1, aggtime2, in_size, out_size))
