from __future__ import absolute_import, division
import numpy as np
from toolz import memoize

from datashader.glyphs.glyph import Glyph
from datashader.utils import isreal, ngjit


class _PointLike(Glyph):
    """Shared methods between Point and Line"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def ndims(self):
        return 1

    @property
    def inputs(self):
        return (self.x, self.y)

    def validate(self, in_dshape):
        if not isreal(in_dshape.measure[str(self.x)]):
            raise ValueError('x must be real')
        elif not isreal(in_dshape.measure[str(self.y)]):
            raise ValueError('y must be real')

    @property
    def x_label(self):
        return self.x

    @property
    def y_label(self):
        return self.y

    def required_columns(self):
        return [self.x, self.y]

    def compute_x_bounds(self, df):
        bounds = self._compute_x_bounds(df[self.x].values)
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds = self._compute_y_bounds(df[self.y].values)
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):

        r = ddf.map_partitions(lambda df: np.array([[
            np.nanmin(df[self.x].values),
            np.nanmax(df[self.x].values),
            np.nanmin(df[self.y].values),
            np.nanmax(df[self.y].values)]]
        )).compute()

        x_extents = np.nanmin(r[:, 0]), np.nanmax(r[:, 1])
        y_extents = np.nanmin(r[:, 2]), np.nanmax(r[:, 3])

        return (self.maybe_expand_bounds(x_extents),
                self.maybe_expand_bounds(y_extents))


class Point(_PointLike):
    """A point, with center at ``x`` and ``y``.

    Points map each record to a single bin.
    Points falling exactly on the upper bounds are treated as a special case,
    mapping into the previous bin rather than being cropped off.

    Parameters
    ----------
    x, y : str
        Column names for the x and y coordinates of each point.
    """
    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        x_name = self.x
        y_name = self.y

        @ngjit
        @self.expand_aggs_and_cols(append)
        def _extend(vt, bounds, xs, ys, *aggs_and_cols):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds

            def map_onto_pixel(x, y):
                """Map points onto pixel grid.

                Points falling on upper bound are mapped into previous bin.
                """
                xx = int(x_mapper(x) * sx + tx)
                yy = int(y_mapper(y) * sy + ty)
                return (xx - 1 if x == xmax else xx,
                        yy - 1 if y == ymax else yy)

            for i in range(xs.shape[0]):
                x = xs[i]
                y = ys[i]
                # points outside bounds are dropped; remainder
                # are mapped onto pixels
                if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                    xi, yi = map_onto_pixel(x, y)
                    append(i, xi, yi, *aggs_and_cols)

        def extend(aggs, df, vt, bounds):
            xs = df[x_name].values
            ys = df[y_name].values
            cols = aggs + info(df)
            _extend(vt, bounds, xs, ys, *cols)

        return extend
