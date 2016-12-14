#!/bin/sh
# Usage:
#    conda-env create -f filetimes.yml
#    source activate filetimes
#    pip install --upgrade git+git://github.com/dask/dask.git@4ae1a203860bd67f09057778c0da5af533b9fcc6
#    pip install --upgrade git+git://github.com/dask/fastparquet.git@ea1680d99ee664905b1c006117411ac4e8a9dae0
#    pip install --upgrade git+git://github.com/numba/numba.git@c80e0a3dfe86c99c474a5fbe7f657b0bab26ada2
#    pip install --upgrade git+git://github.com/blaze/castra.git@1ae53dfcafdd469f0df4620172bf7f6dffb3d5dd
#    pip install --upgrade git+git://github.com/bokeh/datashader.git@f69047ebb762431e14ef87ba43272e5ea0860d0f
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
${timer} python filetimes.py ${1}.castra      dask    census meterswest metersnorth race
#${timer} python filetimes.py ${1}.castra      pandas  census meterswest metersnorth race
${timer} python filetimes.py ${1}.bcolz       dask    census meterswest metersnorth race
${timer} python filetimes.py ${1}.h5          dask    census meterswest metersnorth race
${timer} python filetimes.py ${1}.h5          pandas  census meterswest metersnorth race
${timer} python filetimes.py ${1}.csv         dask    census meterswest metersnorth race
${timer} python filetimes.py ${1}.csv         pandas  census meterswest metersnorth race
#${timer} python filetimes.py ${1}.feather     pandas  census meterswest metersnorth race
