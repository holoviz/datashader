# For use with pyct (https://github.com/pyviz/pyct), but just standard
# pytest config (works with pytest alone).

import os
import pytest

# TODO: want to replace these with nbs using tiny data; just using
# quick and small temporarily to get things going

quick = set([
    'getting_started/2_Pipeline.ipynb',
])

small = set([
    'topics/uk_researchers.ipynb'
])


# ipynb in examples are "example_notebook"; they can additionally be
# "quick" and/or "small"
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
        if os.path.relpath(path,commonpath([os.path.dirname(__file__),path])).replace(os.path.sep,"/") in small:
            item.add_marker(pytest.mark.small)
