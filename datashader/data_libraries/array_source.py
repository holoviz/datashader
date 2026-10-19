from __future__ import annotations

import numpy as np

from datashader import utils
from datashader.compiler import compile_components
from datashader.core import bypixel
from datashader.glyphs import (
    AreaToLineAxis0, AreaToLineAxis0Multi, AreaToZeroAxis0, AreaToZeroAxis0Multi, LineAxis0,
    LineAxis0Multi,
)
from datashader.utils import _ArraySource, _column_with_nan

__all__ = ()

# Glyphs that join consecutive rows, so a block also needs the previous row
_CONNECTED_GLYPHS = (LineAxis0, AreaToZeroAxis0, AreaToLineAxis0)
_MULTI_CONNECTED_GLYPHS = (LineAxis0Multi, AreaToZeroAxis0Multi, AreaToLineAxis0Multi)


def _starts_new_part(source, glyph, row):
    """Return whether the segment from ``row`` starts a new part of the line
    or area, which is when the row before it is missing a coordinate."""
    if row == 0:
        return True
    previous = source._slice(row - 1, row)
    return any(np.isnan(_column_with_nan(previous[col])[0]) for col in glyph.required_columns())


@bypixel.pipeline.register(_ArraySource)
def array_source_pipeline(source, schema, canvas, glyph, summary, *, antialias=False):
    """Aggregate a dict source in blocks of rows into the same aggregates.

    Each block only materializes its own values and masks, which bounds the
    memory used for chunked Arrow columns and for unpacked validity bitmaps.
    """
    create, info, append, _, finalize, antialias_stage_2, antialias_stage_2_funcs, _ = \
        compile_components(summary, schema, glyph, antialias=antialias, cuda=False,
                           partitioned=False)
    extend = glyph._build_extend(
        canvas.x_axis.mapper, canvas.y_axis.mapper, info, append, antialias_stage_2,
        antialias_stage_2_funcs)

    x_range = canvas.x_range or glyph.compute_x_bounds(source)
    y_range = canvas.y_range or glyph.compute_y_bounds(source)
    canvas.validate_ranges(x_range, y_range)

    width = canvas.plot_width
    height = canvas.plot_height

    x_st = canvas.x_axis.compute_scale_and_translate(x_range, width)
    y_st = canvas.y_axis.compute_scale_and_translate(y_range, height)

    x_axis = canvas.x_axis.compute_index(x_st, width)
    y_axis = canvas.y_axis.compute_index(y_st, height)

    bases = create((height, width))

    # Antialiased lines read neighbouring rows and combine whole lines within
    # one extend call. Multi glyphs draw column by column, so blocks would
    # change which row a first or last reduction sees, and where each line
    # starts differs per column.
    if antialias or isinstance(glyph, _MULTI_CONNECTED_GLYPHS):
        block_size = max(len(source), 1)
    else:
        block_size = utils._BLOCK_SIZE
    connected = isinstance(glyph, _CONNECTED_GLYPHS)
    for start, stop in source._block_bounds(block_size):
        if connected and start > 0:
            extend(bases, source._slice(start - 1, stop), x_st + y_st, x_range + y_range,
                   plot_start=_starts_new_part(source, glyph, start - 1))
        else:
            extend(bases, source._slice(start, stop), x_st + y_st, x_range + y_range)

    return finalize(bases,
                    cuda=False,
                    coords=dict([(glyph.x_label, x_axis),
                                 (glyph.y_label, y_axis)]),
                    dims=[glyph.y_label, glyph.x_label],
                    attrs=dict(x_range=x_range, y_range=y_range))
