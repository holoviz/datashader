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
from collections import OrderedDict
from dask.cache import Cache
Cache(9e9).register

base,x,y='data','x','y'
dftype='pandas'
categories=[]

filetypes_storing_categories = {'parq','castra'} # Should also be 


read = OrderedDict(csv={}, h5={}, castra={}, bcolz={}, parq={}, feather={})

read["csv"]     ["pandas"] = lambda filepath,base:  pd.read_csv(filepath)
read["csv"]     ["dask"]   = lambda filepath,base:  dd.read_csv(filepath)
read["h5"]      ["dask"]   = lambda filepath,base:  dd.read_hdf(filepath, base)
read["h5"]      ["pandas"] = lambda filepath,base:  pd.read_hdf(filepath, base)
read["castra"]  ["dask"]   = lambda filepath,base:  dd.from_castra(filepath)
read["bcolz"]   ["dask"]   = lambda filepath,base:  dd.from_bcolz(filepath, chunksize=1000000)
read["feather"] ["pandas"] = lambda filepath,base:  feather.read_dataframe(filepath)
read["parq"]    ["pandas"] = lambda filepath,base:  fp.ParquetFile(filepath).to_pandas()
read["parq"]    ["dask"]   = lambda filepath,base:  dd.io.parquet.read_parquet(filepath,index='index', categories=categories)


write = OrderedDict()

write["csv"]          = lambda df,filepath,base:  df.to_csv(filepath)
write["h5"]           = lambda df,filepath,base:  df.to_hdf(filepath,key=base,format='table')
write["castra"]       = lambda df,filepath,base:  Castra(filepath, template=df,categories=categories).extend(df)
write["bcolz"]        = lambda df,filepath,base:  bcolz.ctable.fromdataframe(df, rootdir=filepath)
write["feather"]      = lambda df,filepath,base:  feather.write_dataframe(df, filepath)
write["parq"]         = lambda df,filepath,base:  fp.write(filepath, df, file_scheme='hive')
write["snappy.parq"]  = lambda df,filepath,base:  fp.write(filepath, df, file_scheme='hive', compression='SNAPPY')
write["gz.parq"]      = lambda df,filepath,base:  fp.write(filepath, df, file_scheme='hive', compression='GZIP')


def timed_write(filepath,output_directory="times"):
    """Accepts any file with a dataframe readable by pandas, and writes it out as a variety of file types"""
    df,duration=timed_read(filepath,"pandas")

    for ext in write.keys():
        directory,filename = os.path.split(filepath)
        basename, extension = os.path.splitext(filename)
        fname = output_directory+os.path.sep+basename+"."+ext
        if os.path.exists(fname):
            print("Keeping existing "+fname)
        else:
            filetype=ext.split(".")[-1]
            if not filetype in filetypes_storing_categories:
                for c in categories:
                    df[c]=df[c].astype(str)
            
            start = time.time()
            write[ext](df,fname,base)
            end = time.time()
            print("{:28} {:05.2f}".format(fname,end-start))
 
            if not filetype in filetypes_storing_categories:
                for c in categories:
                    df[c]=df[c].astype('category')

        
def timed_read(filepath,dftype):
    basename, extension = os.path.splitext(filepath)
    extension = extension[1:]
    code = read[extension].get(dftype,None)
    if code is None:
        return (None,None)
    start = time.time()
    df = code(filepath,base)
    for c in categories:
        df[c]=df[c].astype('category')
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
