from __future__ import absolute_import, division, print_function

"""
Declarative interface to Datashader.

Provides a configurable pipeline that makes it simpler to specify
individual stages independently from the others, Currently does not
add any new underlying capabilities, with all functionality also
available using the default imperative interface provided by other
files.
"""

import param

from . import transfer_functions as tf
from . import reductions
from . import core


class Interpolate(param.Parameterized):
    """
    Parameterized function object to interpolate colors from a scalar input.
    """

    low = param.Parameter(default="lightpink", doc="""
        Color string or tuple specifying the starting point for interpolation.""")

    high = param.Parameter(default="red", doc="""
        Color string or tuple specifying the ending point for interpolation.""")

    how = param.Parameter(default="log", doc="""
        Function object or string specifying how to map a scalar into color space.""")

    def __call__(self, agg):
        return tf.interpolate(agg, self.low, self.high, self.how)


class DatashaderPipeline(param.Parameterized):
    """
    Configurable datashading pipeline.  Allows each element of the pipeline
    to be specified independently without code duplication, e.g. to show the
    effect of varying that element while keeping the rest of the pipeline
    constant.

    The supported pipeline is roughly:

    1. create canvas of the requested size (in data space) and resolution
    2. aggregate using the specified x and y fields, aggregate field, and agg_fn
    3. apply specified transfer_fns, if any, in order
    4. apply specified color_fn to translate each resulting aggregate into a color
    5. return the result as an image
    """

    data = param.Parameter(doc="""
        Object supporting columnar-style data access, such as a Pandas
        dataframe.""")

    x = param.String("x", doc="""
        Name of the field in the supplied data object to use for the x coordinate.""")

    y = param.String("y", doc="""
        Name of the field in the supplied data object to use for the y coordinate.""")

    agg = param.String("count", doc="""
        Name of the field in the supplied data object to use for aggregation.""")

    agg_fn = param.Callable(reductions.count, doc="""
        Function for aggregating pixel-bin contents into a scalar.""")

    transfer_fns = param.HookList(default=[], doc="""
        Optional function(s) to apply to the aggregate after it has been created
        and before it has been converted into a color.""")

    color_fn = param.Callable(default=Interpolate(), doc="""
        Function to convert a scalar value into a color.""")


    def __call__(self, plot, ranges, **params):
        """
        Accepts a Bokeh plot and a viewport specified via a ranges dictionary
        (which should contain x_range, y_range, h, and w).  Returns an image
        rendered at the specified location, using the current parameter values.
        """
        ps = param.ParamOverrides(self,params)

        x_range, y_range = ranges['x_range'], ranges['y_range']
        h, w = ranges['h'], ranges['w']

        cvs = core.Canvas(plot_width=w, plot_height=h,
                          x_range=x_range, y_range=y_range)

        agg = cvs.points(ps.data, ps.x, ps.y, ps.agg_fn(ps.agg))
        for f in ps.transfer_fns:
            agg = f(agg)
        pix = ps.color_fn(agg)

        dh = y_range[1] - y_range[0]
        dw = x_range[1] - x_range[0]
        plot.image_rgba(image=[pix.img], x=x_range[0], y=y_range[0],
                        dw=dw, dh=dh, dilate=False)
