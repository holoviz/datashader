import re
from functools import total_ordering
import numpy as np
from pandas.core.dtypes.dtypes import register_extension_dtype

from datashader.geom.base import Geom, GeomDtype, GeomArray


try:
    # See if we can register extension type with dask >= 1.1.0
    from dask.dataframe.extensions import make_array_nonempty
except ImportError:
    make_array_nonempty = None


@total_ordering
class Points(Geom):
    @classmethod
    def _shapely_to_array_parts(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, (sg.Point, sg.MultiPoint)):
            # Single line
            return [np.asarray(shape.ctypes)]
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must be an instance of Point,
or MultiPoint""".format(typ=type(shape).__name__))

    def to_shapely(self):
        import shapely.geometry as sg
        if len(self.array) == 2:
            return sg.Point(self.array)
        else:
            return sg.MultiPoint(self.array.reshape(len(self.array) // 2, 2))

    @property
    def length(self):
        return 0.0

    @property
    def area(self):
        return 0.0


@register_extension_dtype
class PointsDtype(GeomDtype):
    _type_name = "Points"
    _subtype_re = re.compile(r"^points\[(?P<subtype>\w+)\]$")

    @classmethod
    def construct_array_type(cls):
        return PointsArray


class PointsArray(GeomArray):
    _element_type = Points

    @property
    def _dtype_class(self):
        return PointsDtype

    @property
    def length(self):
        return np.zeros(self.start_indices.shape, dtype=self.flat_array.dtype)

    @property
    def area(self):
        return np.zeros(self.start_indices.shape, dtype=self.flat_array.dtype)


def points_array_non_empty(dtype):
    return PointsArray([[1, 0, 0, 0], [1, 2, 0, 0]], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(PointsDtype)(points_array_non_empty)
