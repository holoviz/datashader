# Temporary script to allow checking notebooks run without errors, or
# to approximately lint check notebooks.
#
# Note: lint checking will not yet work on windows unless sed is
# present or we replace sed with python equivalent.
#
# Run all notebooks & render to html:
#  python examples/test_notebooks.py
#
# Approximately lint check all notebooks:
#  python examples/test_notebooks.py lint

from __future__ import print_function

import sys
import os
import glob
import pprint

############################################################
# Set nbdir, run_skip, and run_allow_error for your project.
# You may need to increase run_cell_timeout if you have
# notebook cells that take a long time to execute.

nbdir = "examples"

run_skip = [
    "examples/census-hv-dask.ipynb",
    "examples/census-hv.ipynb",
    "examples/census.ipynb",
    "examples/edge_bundling.ipynb",
    "examples/holoviews_datashader.ipynb",
    "examples/hover_layer.ipynb",
    "examples/landsat.ipynb",
    "examples/legends.ipynb",
    "examples/lidar.ipynb",
    "examples/lines_vs_points.ipynb",
    "examples/nyc_taxi-nongeo.ipynb",
    "examples/nyc_taxi-paramnb.ipynb",
    "examples/nyc_taxi.ipynb",
    "examples/opensky.ipynb",
    "examples/osm-1billion.ipynb",
    "examples/osm.ipynb",
    "examples/packet_capture_graph.ipynb",
    "examples/plotting_pitfalls.ipynb",     # requires holoviews
    "examples/race_elevation.ipynb",
    "examples/solar.ipynb"
]

run_allow_error = []

run_cell_timeout = 360

############################################################


notebooks = sorted([x.replace(os.path.sep,"/") for x in glob.glob(nbdir+"/*.ipynb")])

checked = []
errored = []
run_skipped = []

if len(sys.argv) == 1:
    do_what = "run"
elif sys.argv[1] == "lint":
    do_what = "lint"
else:
    raise

if do_what=="run":
    for nb in notebooks:
        cmd = "jupyter nbconvert %s --execute --ExecutePreprocessor.kernel_name=python%s --ExecutePreprocessor.timeout=%s --to html"%(nb,sys.version_info[0],run_cell_timeout)
        if nb in run_skip:
            run_skipped.append(nb)
            continue
            
        if nb in run_allow_error:
            cmd += " --allow-errors"
        print(cmd)
        r = os.system(cmd)
        checked.append(nb)
        if r!=0:
            errored.append(nb)

elif sys.argv[1]=='lint':
    for nb in notebooks:
        cmd = """sed -e 's/%/#%/' {f} > {f}~ && jupyter nbconvert {f}~ --to python --PythonExporter.file_extension=.py~ && flake8 --ignore=E,W {p}""".format(f=nb,p=nb[0:-5]+'py~')
        print(cmd)
        r = os.system(cmd)
        checked.append(nb)
        if r!=0:
            errored.append(nb)
else:
    raise

print("%s checked"%len(checked))
if len(checked)>0: pprint.pprint(checked)
print()
print("%s error(s)"%len(errored))
if len(errored)>0: pprint.pprint(errored)
print()

if do_what == 'run':
    print("%s skipped"%len(run_skipped))
    if len(run_skipped)>0: pprint.pprint(run_skipped)
    print()
    if len(run_allow_error) > 0:
        print("Note: the following notebooks were not checked for run errors:")
        pprint.pprint(run_allow_error)

sys.exit(len(errored))
