from toolz import memoize
import numpy as np

from datashader.glyphs.line import _build_map_onto_pixel_for_line
from datashader.glyphs.points import _GeometryLike
from datashader.utils import ngjit

try:
    import spatialpandas
except Exception:
    spatialpandas = None


class PolygonGeom(_GeometryLike):
    # spatialpandas must be available if a PolygonGeom object is created.

    @property
    def geom_dtypes(self):
        from spatialpandas.geometry import PolygonDtype, MultiPolygonDtype
        return PolygonDtype, MultiPolygonDtype

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_polygon = _build_draw_polygon(
            append, map_onto_pixel, x_mapper, y_mapper, expand_aggs_and_cols
        )

        perform_extend_cpu = _build_extend_polygon_geometry(
            draw_polygon, expand_aggs_and_cols
        )
        geom_name = self.geometry

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            geom_array = df[geom_name].array

            perform_extend_cpu(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                geom_array, *aggs_and_cols
            )

        return extend


def _build_draw_polygon(append, map_onto_pixel, x_mapper, y_mapper, expand_aggs_and_cols):
    @ngjit
    @expand_aggs_and_cols
    def draw_polygon(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            offsets, values, xs, ys, yincreasing, eligible,
            *aggs_and_cols
    ):
        """Draw a polygon using a winding-number scan-line algorithm
        """
        # Initialize values of pre-allocated buffers
        xs.fill(np.nan)
        ys.fill(np.nan)
        yincreasing.fill(0)
        eligible.fill(1)

        # First pass, compute bounding box of polygon vertices in data coordinates
        start_index = offsets[0]
        stop_index = offsets[-1]
        # num_edges = stop_index - start_index - 2
        poly_xmin = np.min(values[start_index:stop_index:2])
        poly_ymin = np.min(values[start_index + 1:stop_index:2])
        poly_xmax = np.max(values[start_index:stop_index:2])
        poly_ymax = np.max(values[start_index + 1:stop_index:2])

        # skip polygon if outside viewport
        if (poly_xmax < xmin or poly_xmin > xmax
                or poly_ymax < ymin or poly_ymin > ymax):
            return

        # Compute pixel bounds for polygon
        startxi, startyi = map_onto_pixel(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            max(poly_xmin, xmin), max(poly_ymin, ymin)
        )
        stopxi, stopyi = map_onto_pixel(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            min(poly_xmax, xmax), min(poly_ymax, ymax)
        )
        stopxi += 1
        stopyi += 1

        # Handle subpixel polygons (pixel width and/or height of polygon is 1)
        if (stopxi - startxi) == 1 and (stopyi - startyi) == 1:
            append(i, startxi, startyi, *aggs_and_cols)
            return
        elif (stopxi - startxi) == 1:
            for yi in range(min(startyi, stopyi) + 1, max(startyi, stopyi)):
                append(i, startxi, yi, *aggs_and_cols)
            return
        elif (stopyi - startyi) == 1:
            for xi in range(min(startxi, stopxi) + 1, max(startxi, stopxi)):
                append(i, xi, startyi, *aggs_and_cols)
            return

        # Build arrays of edges in canvas coordinates
        ei = 0
        for j in range(len(offsets) - 1):
            start = offsets[j]
            stop = offsets[j + 1]
            for k in range(start, stop - 2, 2):
                x0 = values[k]
                y0 = values[k + 1]
                x1 = values[k + 2]
                y1 = values[k + 3]

                # Map to canvas coordinates without rounding
                x0c = x_mapper(x0) * sx + tx - 0.5
                y0c = y_mapper(y0) * sy + ty - 0.5
                x1c = x_mapper(x1) * sx + tx - 0.5
                y1c = y_mapper(y1) * sy + ty - 0.5

                if y1c > y0c:
                    xs[ei, 0] = x0c
                    ys[ei, 0] = y0c
                    xs[ei, 1] = x1c
                    ys[ei, 1] = y1c
                    yincreasing[ei] = 1
                elif y1c < y0c:
                    xs[ei, 1] = x0c
                    ys[ei, 1] = y0c
                    xs[ei, 0] = x1c
                    ys[ei, 0] = y1c
                    yincreasing[ei] = -1
                else:
                    # Skip horizontal edges
                    continue

                ei += 1

        # Perform scan-line algorithm
        num_edges = ei
        for yi in range(startyi, stopyi):
            # All edges eligible at start of new row
            eligible.fill(1)
            for xi in range(startxi, stopxi):
                # Init winding number
                winding_number = 0
                for ei in range(num_edges):
                    if eligible[ei] == 0:
                        # We've already determined that edge is above, below, or left
                        # of edge for the current pixel
                        continue

                    # Get edge coordinates.
                    # Note: y1c > y0c due to how xs/ys were populated
                    x0c = xs[ei, 0]
                    x1c = xs[ei, 1]
                    y0c = ys[ei, 0]
                    y1c = ys[ei, 1]

                    # Reject edges that are above, below, or left of current pixel.
                    # Note: Edge skipped if lower vertex overlaps,
                    #       but is kept if upper vertex overlaps
                    if (y0c >= yi or y1c < yi
                            or (x0c < xi and x1c < xi)
                    ):
                        # Edge not eligible for any remaining pixel in this row
                        eligible[ei] = 0
                        continue

                    if xi <= x0c and xi <= x1c:
                        # Edge is fully to the right of the pixel, so we know ray to the
                        # the right of pixel intersects edge.
                        winding_number += yincreasing[ei]
                    else:
                        # Now check if edge is to the right of pixel using cross product
                        # A is vector from pixel to first vertex
                        ax = x0c - xi
                        ay = y0c - yi

                        # B is vector from pixel to second vertex
                        bx = x1c - xi
                        by = y1c - yi

                        # Compute cross product of B and A
                        bxa = (bx * ay - by * ax)

                        if bxa < 0 or (bxa == 0 and yincreasing[ei]):
                            # Edge to the right
                            winding_number += yincreasing[ei]
                        else:
                            # Edge to left, not eligible for any remaining pixel in row
                            eligible[ei] = 0
                            continue

                if winding_number != 0:
                    # If winding number is not zero, point
                    # is inside polygon
                    append(i, xi, yi, *aggs_and_cols)

    return draw_polygon


def _build_extend_polygon_geometry(
        draw_polygon, expand_aggs_and_cols
):
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, geometry, *aggs_and_cols
    ):
        values = geometry.buffer_values
        missing = geometry.isna()
        offsets = geometry.buffer_offsets

        if geometry._sindex is not None:
            # Compute indices of potentially intersecting polygons using
            # geometry's R-tree if there is one
            eligible_inds = geometry.sindex.intersects((xmin, ymin, xmax, ymax))
        else:
            # Otherwise, process all indices
            eligible_inds = np.arange(0, len(geometry), dtype='uint32')

        if len(offsets) == 3:
            # MultiPolygonArray
            offsets0, offsets1, offsets2 = offsets
        else:
            # PolygonArray
            offsets1, offsets2 = offsets
            offsets0 = np.arange(len(offsets1))

        extend_cpu_numba(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            values, missing, offsets0, offsets1, offsets2, eligible_inds, *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            values, missing, offsets0, offsets1, offsets2,
            eligible_inds, *aggs_and_cols
    ):
        # Pre-allocate temp arrays
        max_edges = 0
        if len(offsets0) > 1:
            for i in eligible_inds:
                if missing[i]:
                    continue

                polygon_inds = offsets1[offsets0[i]:offsets0[i + 1] + 1]
                for j in range(len(polygon_inds) - 1):
                    start = offsets2[polygon_inds[j]]
                    stop = offsets2[polygon_inds[j + 1]]
                    max_edges = max(max_edges, (stop - start - 2) // 2)

        xs = np.full((max_edges, 2), np.nan, dtype=np.float32)
        ys = np.full((max_edges, 2), np.nan, dtype=np.float32)
        yincreasing = np.zeros(max_edges, dtype=np.int8)

        # Initialize array indicating which edges are still eligible for processing
        eligible = np.ones(max_edges, dtype=np.int8)

        for i in eligible_inds:
            if missing[i]:
                continue

            polygon_inds = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            for j in range(len(polygon_inds) - 1):
                start = polygon_inds[j]
                stop = polygon_inds[j + 1]

                draw_polygon(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                             offsets2[start:stop + 1], values,
                             xs, ys, yincreasing, eligible, *aggs_and_cols)

    return extend_cpu
