import os
import pytest

quick = set([
    'getting_started/2_Pipeline.ipynb',
#    'user_guide/3_Timeseries.ipynb'
])

# ipynb in examples are "example_notebook"; they can additionally be "quick" 
def pytest_collection_modifyitems(items):
    for item in items:
        path = str(item.fspath)
        if os.path.splitext(path)[1].lower() == ".ipynb":
            item.add_marker(pytest.mark.example_notebook)
        if os.path.relpath(path,os.path.commonpath([os.path.dirname(__file__),path])).replace(os.path.sep,"/") in quick:
            item.add_marker(pytest.mark.quick)
