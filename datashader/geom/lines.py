from __future__ import absolute_import, division
import re
from math import sqrt, isfinite
from functools import total_ordering
import numpy as np

from pandas.core.dtypes.dtypes import register_extension_dtype

from datashader.geom.base import Geom, GeomDtype, GeomArray, _geom_map
from datashader.utils import ngjit


try:
    # See if we can register extension type with dask >= 1.1.0
    from dask.dataframe.extensions import make_array_nonempty
except ImportError:
    make_array_nonempty = None


@total_ordering
class Lines(Geom):
    @classmethod
    def _shapely_to_array_parts(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, (sg.LineString, sg.LinearRing)):
            # Single line
            return [np.asarray(shape.ctypes)]
        elif isinstance(shape, sg.MultiLineString):
            shape = list(shape)
            line_parts = [np.asarray(shape[0].ctypes)]
            line_separator = np.array([np.inf, np.inf])
            for line in shape[1:]:
                line_parts.append(line_separator)
                line_parts.append(np.asarray(line.ctypes))
            return line_parts
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must be an instance of LineString,
MultiLineString, or LinearRing""".format(typ=type(shape).__name__))

    def to_shapely(self):
        import shapely.geometry as sg
        line_breaks = np.concatenate(
            [[-2], np.nonzero(~np.isfinite(self.array))[0][0::2], [len(self.array)]]
        )
        line_arrays = [self.array[start + 2:stop]
                       for start, stop in zip(line_breaks[:-1], line_breaks[1:])]

        lines = [sg.LineString(line_array.reshape(len(line_array) // 2, 2))
                 for line_array in line_arrays]

        if len(lines) == 1:
            return lines[0]
        else:
            return sg.MultiLineString(lines)

    @property
    def length(self):
        return compute_length(self.array)

    @property
    def area(self):
        return 0.0


@register_extension_dtype
class LinesDtype(GeomDtype):
    _type_name = "Lines"
    _subtype_re = re.compile(r"^lines\[(?P<subtype>\w+)\]$")

    @classmethod
    def construct_array_type(cls):
        return LinesArray


class LinesArray(GeomArray):
    _element_type = Lines

    @property
    def _dtype_class(self):
        return LinesDtype

    @property
    def length(self):
        result = np.zeros(self.start_indices.shape, dtype=self.flat_array.dtype)
        _geom_map(self.start_indices, self.flat_array, result, compute_length)
        return result

    @property
    def area(self):
        return np.zeros(self.start_indices.shape, dtype=self.flat_array.dtype)


@ngjit
def compute_length(values):
    total_len = 0.0
    x0 = values[0]
    y0 = values[1]
    for i in range(2, len(values), 2):
        x1 = values[i]
        y1 = values[i+1]

        if isfinite(x0) and isfinite(y0) and isfinite(x1) and isfinite(y1):
            total_len += sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        x0 = x1
        y0 = y1

    return total_len


def lines_array_non_empty(dtype):
    return LinesArray([[1, 0, 1, 1], [1, 2, 0, 0]], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(LinesDtype)(lines_array_non_empty)
