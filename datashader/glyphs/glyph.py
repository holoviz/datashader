from __future__ import annotations
from packaging.version import Version
import inspect
import warnings
import os
from math import isnan

import numpy as np
import pandas as pd

from datashader.utils import Expr, ngjit
from datashader.macros import expand_varargs

try:
    import cudf
    import cupy as cp
except Exception:
    cudf = None
    cp = None


class Glyph(Expr):
    """Base class for glyphs."""

    antialiased = False

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
    def _compute_bounds(s):
        if cudf and isinstance(s, cudf.Series):
            s = s.nans_to_nulls()
            return (s.min(), s.max())
        elif isinstance(s, pd.Series):
            return Glyph._compute_bounds_numba(s.values)
        else:
            return Glyph._compute_bounds_numba(s)

    @staticmethod
    @ngjit
    def _compute_bounds_numba(arr):
        minval = np.inf
        maxval = -np.inf
        for x in arr:
            if not isnan(x):
                if x < minval:
                    minval = x
                if x > maxval:
                    maxval = x

        return minval, maxval

    @staticmethod
    @ngjit
    def _compute_bounds_2d(vals):
        minval = np.inf
        maxval = -np.inf
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i][j]
                if not np.isnan(v):
                    if v < minval:
                        minval = v
                    if v > maxval:
                        maxval = v

        return minval, maxval

    @staticmethod
    def to_cupy_array(df, columns):
        if isinstance(columns, tuple):
            columns = list(columns)

        # Pandas extracts the column name multiple times, but
        # cuDF only extracts each name a single time. For details, see:
        # https://github.com/holoviz/datashader/pull/1050
        if isinstance(columns, list) and (len(columns) != len(set(columns))):
            return cp.stack([cp.array(df[c]) for c in columns], axis=1)

        if Version(cudf.__version__) >= Version("22.02"):
            return df[columns].to_cupy()
        else:
            if not isinstance(columns, list):
                return df[columns].to_gpu_array()
            return df[columns].as_gpu_matrix()

    def expand_aggs_and_cols(self, append):
        """
        Create a decorator that can be used on functions that accept
        *aggs_and_cols as a variable length argument. The decorator will
        replace *aggs_and_cols with a fixed number of arguments.

        The appropriate fixed number of arguments is calculated from the input
        append function.

        Rationale: When we know the fixed length of a variable length
        argument, replacing it with fixed arguments can help numba better
        optimize the the function.

        If this ever causes problems in the future, this decorator can be
        safely removed without changing the functionality of the decorated
        function.

        Parameters
        ----------
        append: function
            The append function for the current aggregator

        Returns
        -------
        function
            Decorator function
        """
        return self._expand_aggs_and_cols(append, self.ndims, self.antialiased)

    @staticmethod
    def _expand_aggs_and_cols(append, ndims, antialiased):
        if os.environ.get('NUMBA_DISABLE_JIT', None):
            # If the NUMBA_DISABLE_JIT environment is set, then we return an
            # identity decorator (one that return function unchanged).
            #
            # Doing this makes it possible to debug functions that are
            # decorated with @jit and @expand_varargs decorators
            return lambda fn: fn

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Numba keeps original function around as append.py_func
                append_args = inspect.getfullargspec(append.py_func).args
            except (TypeError, AttributeError):
                # Treat append as a normal python function
                append_args = inspect.getfullargspec(append).args

        # Get number of arguments accepted by append
        append_arglen = len(append_args)

        # We will subtract 2 because we always pass in the x and y position
        xy_arglen = 2

        # We will also subtract the number of dimensions in this glyph,
        # because that's how many data index arguments are passed to append
        dim_arglen = (ndims or 0)

        # The remaining arguments are for aggregates and columns
        aggs_and_cols_len = append_arglen - xy_arglen - dim_arglen

        # Antialiased append() calls also take aa_factor argument
        if antialiased:
            aggs_and_cols_len -= 1

        return expand_varargs(aggs_and_cols_len)
