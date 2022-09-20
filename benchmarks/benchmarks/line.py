import datashader as ds
import numpy as np
import pandas as pd


class Line:
    param_names = ("line_count", "line_width", "self_intersect")
    params = ([1000, 10000], [0, 1, 2], [False, True])

    def setup(self, line_count, line_width, self_intersect):
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

    def time_LinesAxis1XConstant(self, line_count, line_width, self_intersect):
        if line_width == 0.0 and not self_intersect:
            raise NotImplementedError  # Same as line_width=0, self_intersect=False

        agg = ds.count(self_intersect=self_intersect)
        self.canvas.line(
            self.df, x=self.x, y=list(self.df.columns), agg=agg, axis=1, line_width=line_width,
        )
