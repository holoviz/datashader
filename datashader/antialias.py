from __future__ import annotations

from datashader.enums import AntialiasCombination
from datashader.utils import (
    nanfirst_in_place, nanlast_in_place, nanmax_in_place,
    nanmin_in_place, nansum_in_place, ngjit, parallel_fill)
from numba import literal_unroll


def two_stage_agg(antialias_stage_2: AntialiasStage2 | None):
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


@ngjit
def _combine_in_place(accum_agg, other_agg, antialias_combination):
    if antialias_combination == AntialiasCombination.MAX:
        nanmax_in_place(accum_agg, other_agg)
    elif antialias_combination == AntialiasCombination.MIN:
        nanmin_in_place(accum_agg, other_agg)
    elif antialias_combination == AntialiasCombination.FIRST:
        nanfirst_in_place(accum_agg, other_agg)
    elif antialias_combination == AntialiasCombination.LAST:
        nanlast_in_place(accum_agg, other_agg)
    else:
        nansum_in_place(accum_agg, other_agg)


@ngjit
def aa_stage_2_accumulate(aggs_and_copies, antialias_combinations):
    k = 0
    # Numba access to heterogeneous tuples is only permitted using literal_unroll.
    for agg_and_copy in literal_unroll(aggs_and_copies):
        _combine_in_place(agg_and_copy[1], agg_and_copy[0], antialias_combinations[k])
        k += 1


@ngjit
def aa_stage_2_clear(aggs_and_copies, antialias_zeroes):
    k = 0
    # Numba access to heterogeneous tuples is only permitted using literal_unroll.
    for agg_and_copy in literal_unroll(aggs_and_copies):
        parallel_fill(agg_and_copy[0], antialias_zeroes[k])
        k += 1


@ngjit
def aa_stage_2_copy_back(aggs_and_copies):
    # Numba access to heterogeneous tuples is only permitted using literal_unroll.
    for agg_and_copy in literal_unroll(aggs_and_copies):
        agg_and_copy[0][:] = agg_and_copy[1][:]
