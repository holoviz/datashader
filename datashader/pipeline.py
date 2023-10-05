from __future__ import annotations

from toolz import identity

from . import transfer_functions as tf
from . import reductions
from . import core


class Pipeline:
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
        ``Image`` object. Default is ``shade``.
    spread_fn : callable, optional
        A callable that takes the output of ``color_fn``, and returns another
        ``Image`` object. Default is ``dynspread``.
    height_scale: float, optional
        Factor by which to scale the provided height
    width_scale: float, optional
        Factor by which to scale the provided width
    """
    def __init__(self, df, glyph, agg=reductions.count(),
                 transform_fn=identity, color_fn=tf.shade, spread_fn=tf.dynspread,
                 width_scale=1.0, height_scale=1.0):
        self.df = df
        self.glyph = glyph
        self.agg = agg
        self.transform_fn = transform_fn
        self.color_fn = color_fn
        self.spread_fn = spread_fn
        self.width_scale = width_scale
        self.height_scale = height_scale

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
        canvas = core.Canvas(plot_width=int(width*self.width_scale),
                             plot_height=int(height*self.height_scale),
                             x_range=x_range, y_range=y_range)
        bins = core.bypixel(self.df, canvas, self.glyph, self.agg,
                            antialias=self.glyph.antialiased)
        img = self.color_fn(self.transform_fn(bins))
        return self.spread_fn(img)
