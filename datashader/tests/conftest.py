import numpy as np
import pytest

CUSTOM_MARKS = {"benchmark", "gpu"}


def pytest_addoption(parser):
    for marker in sorted(CUSTOM_MARKS):
        parser.addoption(
            f"--{marker}",
            action="store_true",
            default=False,
            help=f"Run {marker} related tests",
        )


def pytest_configure(config):
    for marker in sorted(CUSTOM_MARKS):
        config.addinivalue_line("markers", f"{marker}: {marker} test marker")


def pytest_collection_modifyitems(config, items):
    skipped, selected = [], []
    markers = {m for m in CUSTOM_MARKS if config.getoption(f"--{m}")}
    empty = not markers
    for item in items:
        item_marks = set(item.keywords) & CUSTOM_MARKS
        if empty and item_marks:
            skipped.append(item)
        elif empty:
            selected.append(item)
        elif not empty and item_marks == markers:
            selected.append(item)
        else:
            skipped.append(item)

    config.hook.pytest_deselected(items=skipped)
    items[:] = selected


@pytest.fixture
def rng():
    return np.random.default_rng(42)
