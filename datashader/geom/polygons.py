from __future__ import absolute_import
import re
from math import isfinite
from functools import total_ordering

import numpy as np
from pandas.core.dtypes.dtypes import register_extension_dtype

from datashader.geom.base import Geom, GeomDtype, GeomArray, _geom_map
from datashader.geom.lines import compute_length
from datashader.utils import ngjit


try:
    # See if we can register extension type with dask >= 1.1.0
    from dask.dataframe.extensions import make_array_nonempty
except ImportError:
    make_array_nonempty = None


@total_ordering
class Polygons(Geom):
    @staticmethod
    def _polygon_to_array_parts(polygon):
        import shapely.geometry as sg
        shape = sg.polygon.orient(polygon)
        exterior = np.asarray(shape.exterior.ctypes)
        polygon_parts = [exterior]
        hole_separator = np.array([-np.inf, -np.inf])
        for ring in shape.interiors:
            interior = np.asarray(ring.ctypes)
            polygon_parts.append(hole_separator)
            polygon_parts.append(interior)
        return polygon_parts

    @classmethod
    def _shapely_to_array_parts(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, sg.Polygon):
            # Single polygon
            return Polygons._polygon_to_array_parts(shape)
        elif isinstance(shape, sg.MultiPolygon):
            shape = list(shape)
            polygon_parts = Polygons._polygon_to_array_parts(shape[0])
            polygon_separator = np.array([np.inf, np.inf])
            for polygon in shape[1:]:
                polygon_parts.append(polygon_separator)
                polygon_parts.extend(Polygons._polygon_to_array_parts(polygon))
            return polygon_parts
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must be an instance of
shapely.geometry.Polygon or shapely.geometry.MultiPolygon"""
                             .format(typ=type(shape).__name__))

    def to_shapely(self):
        import shapely.geometry as sg
        ring_breaks = np.concatenate(
            [[-2], np.nonzero(~np.isfinite(self.array))[0][0::2], [len(self.array)]]
        )
        polygon_breaks = set(np.concatenate(
            [[-2], np.nonzero(np.isposinf(self.array))[0][0::2], [len(self.array)]]
        ))

        # Build rings for both outer and holds
        rings = []
        for start, stop in zip(ring_breaks[:-1], ring_breaks[1:]):
            ring_array = self.array[start + 2: stop]
            ring_pairs = ring_array.reshape(len(ring_array) // 2, 2)
            rings.append(sg.LinearRing(ring_pairs))

        # Build polygons
        polygons = []
        outer = None
        holes = []
        for ring, start in zip(rings, ring_breaks[:-1]):
            if start in polygon_breaks:
                if outer:
                    # This is the first ring in a new polygon, construct shapely polygon
                    # with already collected rings
                    polygons.append(sg.Polygon(outer, holes))

                # Start collecting new polygon
                outer = ring
                holes = []
            else:
                # Ring is a hole
                holes.append(ring)

        # Build final polygon
        polygons.append(sg.Polygon(outer, holes))

        if len(polygons) == 1:
            return polygons[0]
        else:
            return sg.MultiPolygon(polygons)

    @property
    def length(self):
        return compute_length(self.array)

    @property
    def area(self):
        return compute_area(self.array)


@register_extension_dtype
class PolygonsDtype(GeomDtype):
    _type_name = "Polygons"
    _subtype_re = re.compile(r"^polygons\[(?P<subtype>\w+)\]$")

    @classmethod
    def construct_array_type(cls):
        return PolygonsArray


class PolygonsArray(GeomArray):
    _element_type = Polygons

    @property
    def _dtype_class(self):
        return PolygonsDtype

    @property
    def length(self):
        result = np.zeros(self.start_indices.shape, dtype=self.flat_array.dtype)
        _geom_map(self.start_indices, self.flat_array, result, compute_length)
        return result

    @property
    def area(self):
        result = np.zeros(self.start_indices.shape, dtype=self.flat_array.dtype)
        _geom_map(self.start_indices, self.flat_array, result, compute_area)
        return result


@ngjit
def compute_area(values):
    area = 0.0
    if len(values) < 6:
        # A degenerate polygon
        return 0.0
    polygon_start = 0
    for k in range(0, len(values) - 4, 2):
        i, j = k + 2, k + 4
        ix = values[i]
        jy = values[j + 1]
        ky = values[k + 1]
        if not isfinite(values[j]):
            # last vertex not finite, polygon traversal finished, add wraparound term
            polygon_stop = j
            firstx = values[polygon_start]
            secondy = values[polygon_start + 3]
            lasty = values[polygon_stop - 3]
            area += firstx * (secondy - lasty)
        elif not isfinite(values[i]):
            # middle vertex not finite, but last vertex is.
            # We're going to start a new polygon
            polygon_start = j
        elif isfinite(ix) and isfinite(jy) and isfinite(ky):
            area += ix * (jy - ky)

    # wrap-around term for final polygon
    firstx = values[polygon_start]
    secondy = values[polygon_start + 3]
    lasty = values[len(values) - 3]
    area += firstx * (secondy - lasty)

    return area / 2.0


def polygons_array_non_empty(dtype):
    return PolygonsArray([[1, 0, 0, 0, 2, 2], [1, 2, 0, 0, 2, 2]], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(PolygonsDtype)(polygons_array_non_empty)
