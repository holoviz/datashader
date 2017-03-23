#!/bin/sh
# Usage:
#    conda-env create -f filetimes.yml
#    patch --verbose -b -p4 -d $CONDA_PREFIX/lib/python3.5/site-packages/fastparquet < fastparquet.type_mismatch.patch
#    source activate filetimes
#    mkdir times
#    python -c "import filetimes as ft ; ft.p.base='census' ; ft.p.x='meterswest' ; ft.p.y='metersnorth' ; ft.p.categories=['race']; ft.timed_write('data/tinycensus.csv',dftype='pandas')"
#    # (or 'data/census.h5' and/or dftype='dask')
#    ./filetimes.sh times/tinycensus

timer=/usr/bin/time
timer="" # External timing disabled to avoid unhelpful "Command terminated abnormally" messages

${timer} python filetimes.py ${1}.parq        dask    census meterswest metersnorth race
${timer} python filetimes.py ${1}.snappy.parq dask    census meterswest metersnorth race
${timer} python filetimes.py ${1}.parq        pandas  census meterswest metersnorth race
${timer} python filetimes.py ${1}.snappy.parq pandas  census meterswest metersnorth race
#${timer} python filetimes.py ${1}.castra      dask    census meterswest metersnorth race
${timer} python filetimes.py ${1}.castra      pandas  census meterswest metersnorth race
${timer} python filetimes.py ${1}.bcolz       dask    census meterswest metersnorth race
${timer} python filetimes.py ${1}.h5          dask    census meterswest metersnorth race
${timer} python filetimes.py ${1}.h5          pandas  census meterswest metersnorth race
${timer} python filetimes.py ${1}.csv         dask    census meterswest metersnorth race
${timer} python filetimes.py ${1}.csv         pandas  census meterswest metersnorth race
#${timer} python filetimes.py ${1}.feather     pandas  census meterswest metersnorth race
