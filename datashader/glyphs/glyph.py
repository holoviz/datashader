from __future__ import absolute_import, division
import numpy as np

from datashader.utils import Expr, ngjit


class Glyph(Expr):
    """Base class for glyphs."""

    @property
    def ndims(self):
        """
        The number of dimensions required in the data structure this Glyph is
        constructed from. Or None if input data structure is irregular

        For example
         * ndims is 1 if glyph is constructed from a DataFrame
         * ndims is 2 if glyph is constructed from a 2D xarray DataArray
         * ndims is None if glyph is constructed from multiple DataFrames of
           different lengths
        """
        raise NotImplementedError()

    @staticmethod
    def maybe_expand_bounds(bounds):
        minval, maxval = bounds
        if not (np.isfinite(minval) and np.isfinite(maxval)):
            minval, maxval = -1.0, 1.0
        elif minval == maxval:
            minval, maxval = minval-1, minval+1
        return minval, maxval

    @staticmethod
    @ngjit
    def _compute_x_bounds(xs):
        minval = np.inf
        maxval = -np.inf
        for x in xs:
            if not np.isnan(x):
                if x < minval:
                    minval = x
                if x > maxval:
                    maxval = x

        return minval, maxval

    @staticmethod
    @ngjit
    def _compute_y_bounds(ys):
        minval = np.inf
        maxval = -np.inf
        for y in ys:
            if not np.isnan(y):
                if y < minval:
                    minval = y
                if y > maxval:
                    maxval = y

        return minval, maxval


