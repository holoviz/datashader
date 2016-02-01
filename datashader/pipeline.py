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
from . import glyphs


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
    Configurable datashading pipeline for use with Bokeh.  Allows each
    element of the pipeline to be specified independently without code
    duplication, e.g. to show the effect of varying that element while
    keeping the rest of the pipeline constant.

    Given a dataframe-like object df, the supported pipeline is roughly:

    1. create canvas of the requested size (in data space) and resolution
    2. aggregate df into pixel-shaped bins using the glyph and agg specifications
    3. apply specified transfer_fns, if any, in order, on the set of bin values
    4. apply specified color_fn to translate each resulting bin value into a color
    5. return the result as an image
    """

    df = param.Parameter(doc="""
        Object supporting columnar-style data access, such as a Pandas
        dataframe.""")

    glyph = param.ClassSelector(glyphs.Glyph,default=glyphs.Point("x","y"), doc="""
        Marker shape for each point, specified using fields for the data object.""")

    agg = param.ClassSelector(reductions.Reduction, default=reductions.count("count"), 
        doc="""Function for incrementally reducing a bin's values into a scalar.""")

    transfer_fns = param.HookList(default=[], doc="""
        Optional function(s) to apply to the aggregated bin values, before
        they each get converted into a color.""")

    color_fn = param.Callable(default=Interpolate(), doc="""
        Function to convert a scalar aggregated bin value into a color.""")


    def __call__(self, plot, ranges, **params):
        """
        Accepts a Bokeh plot and a viewport specified via a ranges dictionary
        (which should contain x_range and y_range specifying the viewport in
        data space, and h and w specifying the resolution).

        Returns an image of the specified height and width, rendered
        over the specified range in data space, using the current
        parameter values.
        """
        ps = param.ParamOverrides(self,params)

        x_range, y_range = ranges['x_range'], ranges['y_range']
        h, w = ranges['h'], ranges['w']

        canvas = core.Canvas(plot_width=w, plot_height=h,
                             x_range=x_range, y_range=y_range)

        bins = core.bypixel(ps.df, canvas, ps.glyph, ps.agg)

        for f in ps.transfer_fns:
            agg = f(bins)
        pixels = ps.color_fn(bins)

        dh = y_range[1] - y_range[0]
        dw = x_range[1] - x_range[0]
        plot.image_rgba(image=[pixels.img], x=x_range[0], y=y_range[0],
                        dw=dw, dh=dh, dilate=False)
