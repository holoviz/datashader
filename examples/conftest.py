import pytest

quick = set([
    'datashader/examples/getting_started/2_Pipeline.ipynb',
#    'datashader/examples/user_guide/3_Timeseries.ipynb'
])

# ipynb in examples are "example_notebook"; they can additionally be "quick" 
def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.split("::")[0].startswith("datashader/examples"):
            item.add_marker(pytest.mark.example_notebook)
        # NOTE: not sure where I ought to get name from - is this the
        # right bit of pytest api, and will it work on windows?
        if item.nodeid.split("::")[0] in quick:
            item.add_marker(pytest.mark.quick)
