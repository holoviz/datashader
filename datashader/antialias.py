from __future__ import annotations
from enum import Enum
from typing import NamedTuple, TYPE_CHECKING

from datashader.utils import (
    nanfirst_in_place, nanlast_in_place, nanmax_in_place,
    nanmin_in_place, nansum_in_place, ngjit, parallel_fill)
from numba import literal_unroll


# Enum used to specify how the second stage aggregation is performed
# for 2-stage antialiased lines.
class AntialiasCombination(Enum):
    SUM_1AGG = 1
    SUM_2AGG = 2
    MIN = 3
    MAX = 4
    FIRST = 5
    LAST = 6


class AntialiasStage2(NamedTuple):
    """Configuration for second-stage combination of a single antialiased reduction."""
    combination: AntialiasCombination
    zero: float
    n_reduction: bool = False
    categorical: bool = False


if TYPE_CHECKING:
    UnzippedAntialiasStage2 = tuple[tuple[AntialiasCombination], tuple[float], tuple[bool], tuple[bool]]


def two_stage_agg(antialias_stage_2: UnzippedAntialiasStage2 | None):
    """Information used to perform the correct stage 2 aggregation."""
    if not antialias_stage_2:
        # Not using antialiased lines, doesn't matter what is returned.
        return False, False

    # A single combination in (SUM_2AGG, FIRST, LAST, MIN) means that a 2-stage
    # aggregation will be used, otherwise use a 1-stage aggregation that is
    # faster.
    use_2_stage_agg = False
    for comb in antialias_stage_2[0]:
        if comb in (AntialiasCombination.SUM_2AGG, AntialiasCombination.MIN,
                    AntialiasCombination.FIRST, AntialiasCombination.LAST):
            use_2_stage_agg = True
            break

    # Boolean overwrite flag is used in _full_antialias() is True to overwrite
    # pixel values (using max of previous and new values) or False for the more
    # complicated correction algorithm. Prefer overwrite=True for speed, but
    # any SUM_1AGG implies overwrite=False.
    overwrite = True
    for comb in antialias_stage_2[0]:
        if comb == AntialiasCombination.SUM_1AGG:
            overwrite = False
            break

    return overwrite, use_2_stage_agg
