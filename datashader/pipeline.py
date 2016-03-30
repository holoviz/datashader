from __future__ import absolute_import, division, print_function

from toolz import identity

from . import transfer_functions as tf
from . import reductions
from . import core


class Pipeline(object):
    """A datashading pipeline callback.

    Given a declarative specification, creates a callable with the following
    signature:

    ``callback(x_range, y_range, width, height)``

    where ``x_range`` and ``y_range`` form the bounding box on the viewport,
    and ``width`` and ``height`` specify the output image dimensions.

    Parameters
    ----------
    df : pandas.DataFrame, dask.DataFrame
    glyph : Glyph
        The glyph to bin by.
    agg : Reduction, optional
        The reduction to compute per-pixel. Default is ``count()``.
    transform_fn : callable, optional
        A callable that takes the computed aggregate as an argument, and
        returns another aggregate. This can be used to do preprocessing before
        passing to the ``color_fn`` function.
    color_fn : callable, optional
        A callable that takes the output of ``tranform_fn``, and returns an
        ``Image`` object. Default is ``interpolate``.
    spread_fn : callable, optional
        A callable that takes the output of ``color_fn``, and returns another
        ``Image`` object. Default is ``dynspread``.
    """
    def __init__(self, df, glyph, agg=reductions.count(),
                 transform_fn=identity, color_fn=tf.interpolate,  spread_fn=tf.dynspread):
        self.df = df
        self.glyph = glyph
        self.agg = agg
        self.transform_fn = transform_fn
        self.color_fn = color_fn
        self.spread_fn = spread_fn

    def __call__(self, x_range=None, y_range=None, width=600, height=600):
        """Compute an image from the specified pipeline.

        Parameters
        ----------
        x_range, y_range : tuple, optional
            The bounding box on the viewport, specified as tuples of
            ``(min, max)``
        width, height : int, optional
            The shape of the image
        """
        canvas = core.Canvas(plot_width=width, plot_height=height,
                             x_range=x_range, y_range=y_range)
        bins = core.bypixel(self.df, canvas, self.glyph, self.agg)
        img = self.color_fn(self.transform_fn(bins))
        return self.spread_fn(img)
