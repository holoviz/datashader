#!/bin/sh
# Usage:
#    conda-env create -f filetimes.yml
#    source activate filetimes
#    conda remove dask fastparquet
#    pip install git+git://github.com/dask/dask.git@b6ba65dd788581f3bb589684f4f09c8c92f43607
#    pip install git+git://github.com/dask/fastparquet.git@66c8c445f054e96d4ada41d330385c19880996f1
#    pip install git+git://github.com/blaze/castra.git@1ae53dfcafdd469f0df4620172bf7f6dffb3d5dd
#    mkdir times
#    python -c "import filetimes ; filetimes.base='census' ; filetimes.categories=['race']; filetimes.timed_write('data/tinycensus.csv')" # or census.h5
#    ./filetimes.sh times/tinycensus

/usr/bin/time python filetimes.py ${1}.csv         dask    census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.h5          dask    census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.castra      dask    census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.bcolz       dask    census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.feather     dask    census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.parq        dask    census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.gz.parq     dask    census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.snappy.parq dask    census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'

/usr/bin/time python filetimes.py ${1}.csv         pandas  census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.h5          pandas  census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.castra      pandas  census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.bcolz       pandas  census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.feather     pandas  census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.parq        pandas  census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.gz.parq     pandas  census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
/usr/bin/time python filetimes.py ${1}.snappy.parq pandas  census meterswest metersnorth 2>&1 | tr '\n' ' ' | sed -e 's/ real.*/\n/'
