import os
import pytest

quick = set([
    # relative to conftest.py's location
    'getting_started/2_Pipeline.ipynb',
#    'user_guide/3_Timeseries.ipynb'
])

# ipynb in examples are "example_notebook"; they can additionally be "quick" 
def pytest_collection_modifyitems(items):
    try:
        commonpath = os.path.commonpath
    except AttributeError:
        # for py2...but note: might go wrong
        # (https://docs.python.org/3/library/os.path.html#os.path.commonprefix). Could
        # probably figure out overlap below entirely differently.
        commonpath = os.path.commonprefix

    for item in items:
        path = str(item.fspath)
        if os.path.splitext(path)[1].lower() == ".ipynb":
            item.add_marker(pytest.mark.example_notebook)
        if os.path.relpath(path,commonpath([os.path.dirname(__file__),path])).replace(os.path.sep,"/") in quick:
            item.add_marker(pytest.mark.quick)
