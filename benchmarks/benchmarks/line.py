import dask.dataframe as dd
import datashader as ds
import numpy as np
import pandas as pd
from .common import DataLibrary

try:
    import cudf
except:
    cudf = None

try:
    import dask_cudf
except:
    dask_cudf = None


class Line:
    param_names = ("data_library", "line_count", "line_width", "self_intersect")
    params = (
        [DataLibrary.PandasDF, DataLibrary.DaskDF, DataLibrary.CuDF, DataLibrary.DaskCuDF],
        [1000, 10000], [0, 1, 2], [False, True],
    )

    def setup(self, data_library, line_count, line_width, self_intersect):
        canvas_size = 1000
        points_per_line = 10

        self.canvas = ds.Canvas(canvas_size, canvas_size)

        self.x = np.linspace(0, 1, points_per_line)
        rng = np.random.default_rng(428921)
        y = np.cumsum(
            np.c_[np.zeros((line_count, 1)), rng.standard_normal((line_count, points_per_line))],
            axis=1,
        )
        self.df = pd.DataFrame(y)

        if data_library == DataLibrary.PandasDF:
            pass
        elif data_library == DataLibrary.DaskDF:
            self.df = dd.from_pandas(self.df, npartitions=4)
        elif data_library == DataLibrary.CuDF:
            if cudf:
                self.df = cudf.DataFrame.from_pandas(self.df)
            else:
                raise NotImplementedError("CuDF not available")
        elif data_library == DataLibrary.DaskCuDF:
            if dask_cudf:
                cdf = cudf.DataFrame.from_pandas(self.df)
                self.df = dask_cudf.from_cudf(cdf, npartitions=4)
            else:
                raise NotImplementedError("Dask-cuDF not available")
        else:
            raise NotImplementedError(f"data_library {data_library} not supported in this test")

    def time_LinesAxis1XConstant(self, data_library, line_count, line_width, self_intersect):
        if line_width == 0 and not self_intersect:
            raise NotImplementedError  # Same as line_width=0, self_intersect=False
        elif line_width > 0 and data_library not in [DataLibrary.PandasDF, DataLibrary.DaskDF]:
            raise NotImplementedError  # Antialiased lines only work on CPU not GPU

        agg = ds.count(self_intersect=self_intersect)
        self.canvas.line(
            self.df, x=self.x, y=list(self.df.columns), agg=agg, axis=1, line_width=line_width,
        )
