#!/bin/sh

# Usage:
#    conda env create -f filetimes.yml
#    source activate filetimes
#    conda uninstall --force dask fastparquet python-snappy
#    pip install --no-cache-dir --upgrade git+https://github.com/dask/dask@964b377    # auto-detect categoricals for dd.read_parquet
#    pip install --no-cache-dir --upgrade git+https://github.com/dask/fastparquet@4106c30    # auto-detect categoricals for dd.read_parquet
#    pip install --no-cache-dir git+https://github.com/andrix/python-snappy@0d1ab38    # For releasing the GIL. May need to pin llvmlite to the version that numba depends on
#    mkdir times
#    python -c "import filetimes as ft ; ft.p.base='census' ; ft.p.x='meterswest' ; ft.p.y='metersnorth' ; ft.p.categories=['race']; ft.timed_write('data/tinycensus.csv',dftype='pandas',fsize='double')"
#    # (dftype can also be 'dask', fsize can also be 'single')
#    ./filetimes.sh times/tinycensus
#    # (add a second argument to filetimes.sh to set the caching mode)
#    # (add a third argument to filetimes.sh to set the ft.DEBUG variable)
#
#    More examples of filetimes.sh:
#      1) Use no caching, but enable DEBUG messages:
#             ./filetimes.sh times/tinycensus '' debug
#      2) Use "persist" caching mode:
#             ./filetimes.sh times/tinycensus persist
#      3) Use "cachey" caching mode (force-loads dask dataframes), enable DEBUG messages:
#             ./filetimes.sh times/tinycensus cachey debug

timer=/usr/bin/time
timer="" # External timing disabled to avoid unhelpful "Command terminated abnormally" messages

# Display each command if a third argument is provided
test -n "$3" && set -x

${timer} python filetimes.py ${1}.parq        dask    census meterswest metersnorth race ${3:+--debug} ${2:+--cache=$2}
${timer} python filetimes.py ${1}.snappy.parq dask    census meterswest metersnorth race ${3:+--debug} ${2:+--cache=$2}
${timer} python filetimes.py ${1}.parq        pandas  census meterswest metersnorth race ${3:+--debug} ${2:+--cache=$2}
${timer} python filetimes.py ${1}.snappy.parq pandas  census meterswest metersnorth race ${3:+--debug} ${2:+--cache=$2}
${timer} python filetimes.py ${1}.castra      dask    census meterswest metersnorth race ${3:+--debug} ${2:+--cache=$2}
${timer} python filetimes.py ${1}.castra      pandas  census meterswest metersnorth race ${3:+--debug} ${2:+--cache=$2}
${timer} python filetimes.py ${1}.bcolz       dask    census meterswest metersnorth race ${3:+--debug} ${2:+--cache=$2}
${timer} python filetimes.py ${1}.h5          dask    census meterswest metersnorth race ${3:+--debug} ${2:+--cache=$2}
${timer} python filetimes.py ${1}.h5          pandas  census meterswest metersnorth race ${3:+--debug} ${2:+--cache=$2}
${timer} python filetimes.py ${1}.csv         dask    census meterswest metersnorth race ${3:+--debug} ${2:+--cache=$2}
${timer} python filetimes.py ${1}.csv         pandas  census meterswest metersnorth race ${3:+--debug} ${2:+--cache=$2}
${timer} python filetimes.py ${1}.feather     pandas  census meterswest metersnorth race ${3:+--debug} ${2:+--cache=$2}
