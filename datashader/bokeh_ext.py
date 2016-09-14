from __future__ import absolute_import

import uuid
import json

import numpy as np

from bokeh.io import notebook_div
from bokeh.models import CustomJS, ColumnDataSource, Square, HoverTool, GlyphRenderer
from bokeh.model import _ModelInDocument as add_to_document
from bokeh.io import _CommsHandle
from bokeh.util.notebook import get_comms
from bokeh.models import Plot, Text, Circle, Range1d
from bokeh.plotting import Figure

from . import transfer_functions as tf
from .utils import downsample_aggregate, summarize_aggregate_values


class InteractiveImage(object):
    """
    Bokeh-based interactive image object that updates on pan/zoom
    events.

    Given a Bokeh plot and a callback function, calls the function
    whenever the pan or zoom changes the plot's extent, regenerating
    the image dynamically.  Works in a Jupyter/IPython notebook cell,
    using the existing notebook kernel Python process (not a separate
    Bokeh server).  Does not yet support usage outside the notebook,
    but could be extened to use Bokeh server in that case.

    Parameters
    ----------
    bokeh_plot : plot or figure
        Bokeh plot the image will be drawn on
    callback : function
        Python callback function with the signature::

           fn(x_range=(xmin, xmax), y_range=(ymin, ymax),
              w, h, **kwargs)

        and returning a PIL image object.
    delay : int
        Specifies the delay between the first received event
        and when events are actually processed. Useful to
        ignore events generated immediately after initiating
        zooming.
    timeout: int
        Determines the timeout after which the callback will
        process new events without the previous one having
        reported completion. Increase for very long running
        callbacks.
    **kwargs
        Any kwargs provided here will be passed to the callback
        function.
    """

    jscode = """
        // Define a callback to capture errors on the Python side
        function callback(msg){{
          if (msg.msg_type == "execute_result") {{
            if (msg.content.data['text/plain'] === "'Complete'") {{
               if (Bokeh._queued.length) {{
                   update_plot();
               }} else {{
                   Bokeh._blocked = false;
               }}
               Bokeh._timeout = Date.now();
            }}
          }} else {{
            console.log("Python callback returned unexpected message:", msg)
          }}
        }}
        callbacks = {{iopub: {{output: callback}}}};

        function update_plot() {{
            range = Bokeh._queued;
            var cmd = "{cmd}(" + range + ")"
            // Execute the command on the Python kernel
            if (IPython.notebook.kernel !== undefined) {{
                var kernel = IPython.notebook.kernel;
                kernel.execute(cmd, callbacks, {{silent : false}});
            }}
            Bokeh._queued = [];
        }}

        var plot = x_range.plots[0];
        // Generate a command to execute in Python
        var ranges = {{xmin: x_range.start,
                       ymin: y_range.start,
                       xmax: x_range.end,
                       ymax: y_range.end,
                       w: Math.floor(plot.width),
                       h: Math.floor(plot.height)}}
        var range_str = JSON.stringify(ranges)

        if (!Bokeh._queued) {{
            Bokeh._queued = [];
            Bokeh._blocked = false;
            Bokeh._timeout = Date.now();
        }}

        timeout = Bokeh._timeout + {timeout};
        if (typeof _ === "undefined") {{
        }} else if ((Bokeh._blocked && (Date.now() < timeout))) {{
            Bokeh._queued = [range_str];
        }} else {{
            Bokeh._queued = [range_str];
            setTimeout(update_plot(), {delay});
            Bokeh._blocked = true;
            Bokeh._timeout = Date.now();
        }}
    """

    cmd_template = "from {module} import {cls}; {cls}._callbacks['{ref}'].update"

    _callbacks = {}

    def __init__(self, bokeh_plot, callback, delay=200, timeout=2000, throttle=None,
                 **kwargs):
        self.p = bokeh_plot
        self.callback = callback
        self.kwargs = kwargs
        self.ref = str(uuid.uuid4())
        self.comms_handle = None
        self.delay = delay
        self.timeout = timeout
        if throttle:
            print("Warning: throttle parameter no longer supported; will not be accepted in future versions")

        # Initialize the image and callback
        self.ds, self.renderer = self._init_image()
        callback = self._init_callback()
        self.p.x_range.callback = callback
        self.p.y_range.callback = callback

        # Initialize document
        doc_handler = add_to_document(self.p)
        with doc_handler:
            self.doc = doc_handler._doc
            self.div = notebook_div(self.p, self.ref)

    def _init_callback(self):
        """
        Generate CustomJS from template.
        """
        cls = type(self)

        # Register callback on the class with unique reference
        cls._callbacks[self.ref] = self

        # Generate python callback command
        cmd = cls.cmd_template.format(module=cls.__module__,
                                      cls=cls.__name__, ref=self.ref)

        # Initialize callback
        cb_code = cls.jscode.format(cmd=cmd,
                                    ref=self.ref.replace('-', '_'),
                                    delay=self.delay,
                                    timeout=self.timeout)
        cb_args = dict(x_range=self.p.x_range, y_range=self.p.y_range)
        return CustomJS(args=cb_args, code=cb_code)

    def _init_image(self):
        """
        Initialize RGBA image glyph and datasource
        """
        width, height = self.p.plot_width, self.p.plot_height
        xmin, xmax = self.p.x_range.start, self.p.x_range.end
        ymin, ymax = self.p.y_range.start, self.p.y_range.end

        x_range = (xmin, xmax)
        y_range = (ymin, ymax)
        dw, dh = xmax - xmin, ymax - ymin
        image = self.callback(x_range, y_range, width, height, **self.kwargs)

        ds = ColumnDataSource(data=dict(image=[image.data], x=[xmin],
                                        y=[ymin], dw=[dw], dh=[dh]))
        renderer = self.p.image_rgba(source=ds, image='image', x='x', y='y',
                                     dw='dw', dh='dh', dilate=False)
        return ds, renderer

    def update(self, ranges):
        """
        Update the image datasource based on the new ranges,
        serialize the data to JSON and send to notebook via
        a new or existing notebook comms handle.

        Parameters
        ----------
        ranges : dict(xmin=float, xmax=float, ymin=float, ymax=float,
                      h=int, w=int)
            Dictionary with of x/y-ranges, width and height.
        """
        if not self.comms_handle:
            self.comms_handle = _CommsHandle(get_comms(self.ref),
                                             self.doc, {})

        self.update_image(ranges)
        msg = self.get_update_event()
        self.comms_handle.comms.send(msg)
        return 'Complete'


    def get_update_event(self):
        """
        Generate an update event json message.
        """
        data = dict(self.ds.data)
        data['image'] = [data['image'][0].tolist()]
        return json.dumps({'events': [{'attr': u'data',
                                       'kind': 'ModelChanged',
                                       'model': self.ds.ref,
                                       'new': data}],
                           'references': []})

    def update_image(self, ranges):
        """
        Updates image with data returned by callback
        """
        x_range = (ranges['xmin'], ranges['xmax'])
        y_range = (ranges['ymin'], ranges['ymax'])
        dh = y_range[1] - y_range[0]
        dw = x_range[1] - x_range[0]

        image = self.callback(x_range, y_range, ranges['w'],
                              ranges['h'], **self.kwargs)
        new_data = dict(image=[image.data], x=[x_range[0]],
                        y=[y_range[0]], dw=[dw], dh=[dh])
        self.ds.data.update(new_data)

    def _repr_html_(self):
        return self.div

class HoverLayer(object):
    """
    Wrapper for adding a HoverTool instance to a plot tools which
    highlights values under the user's mouse location.

    Parameters
    ----------
    field_name : str
        Field title which will appear in hover tooltip

    highlight_fill_color : str
        Fill color for glyph which appears on mouse over.

    highlight_line_color : str
        Line color for glyph which appears on mouse over.

    size : int
        Defined hover layer resolution in pixels
        (i.e. height/width of hover grid)

    extent : list
        ``[xmin, ymin, xmax, ymax]`` in data coordinates representing aggregate bounds

    agg : xarray
        Datashader aggregate object (e.g. result of Canvas.points())

    how : str
        Downsample method for summarizing ordinal aggregates (default: mean).
        Options include: mean, sum, max, min, median, std, var, count.
    """

    def __init__(self,
                 field_name='Value',
                 highlight_fill_color='#79DCDE',
                 highlight_line_color='#79DCDE',
                 size=8,
                 is_categorical=False,
                 extent=None,
                 agg=None,
                 how='mean'):

        if how not in ('mean', 'sum', 'max', 'min', 'median', 'std', 'var', 'count'):
            raise ValueError("invalid 'how' downsample method")

        self.hover_data = ColumnDataSource(data=dict(x=[], y=[], value=[]))

        self.invisible_square = Square(x='x',
                                       y='y',
                                       fill_color=None,
                                       line_color=None,
                                       size=size)

        self.visible_square = Square(x='x',
                                     y='y',
                                     fill_color=highlight_fill_color,
                                     fill_alpha=.5,
                                     line_color=highlight_line_color,
                                     line_alpha=1,
                                     size=size)
        self.tooltips = []

        code = "source.selected = cb_data['index'];"
        self._callback = CustomJS(args={'source': self.hover_data}, code=code)

        self.renderer = GlyphRenderer()
        self.renderer.data_source = self.hover_data
        self.renderer.glyph = self.invisible_square
        self.renderer.selection_glyph = self.visible_square
        self.renderer.nonselection_glyph = self.invisible_square

        self.tool = HoverTool(callback=self._callback,
                              renderers=[self.renderer],
                              mode='mouse')

        self.extent = extent
        self.is_categorical = is_categorical
        self.field_name = field_name

        self._agg = agg
        self._size = size or 8

        self.how = how

        if self.agg is not None and self.extent is not None:
            self.compute()

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value
        self.invisible_square.size = value
        self.visible_square.size = value

        if self.agg is not None and self.extent is not None:
            self.compute()

    @property
    def agg(self):
        return self._agg

    @agg.setter
    def agg(self, value):
        self._agg = value

        if self.agg is not None and self.extent is not None:
            self.compute()

    def compute(self):
        sq_xs = np.linspace(self.extent[0],
                            self.extent[2],
                            self.agg.shape[1] / self.size)

        sq_ys = np.linspace(self.extent[1],
                            self.extent[3],
                            self.agg.shape[0] / self.size)

        agg_xs, agg_ys = np.meshgrid(sq_xs, sq_ys)
        self.hover_data.data['x'] = agg_xs.flatten()
        self.hover_data.data['y'] = agg_ys.flatten()
        self.hover_agg = downsample_aggregate(self.agg.values, self.size, how=self.how)

        tooltips = []
        if self.is_categorical:
            cats = self.agg[self.agg.dims[2]].values.tolist()
            for i, e in enumerate(cats):
                self.hover_data.data[str(e)] = self.hover_agg[:, :, i].flatten()
                tooltips.append((str(e), '@{}'.format(str(e))))
        else:
            self.hover_data.data['value'] = self.hover_agg.flatten()
            tooltips.append((self.field_name, '@value'))

        self.tool.tooltips = tooltips
        return self.hover_agg

def create_ramp_legend(agg, cmap, how='linear', width=600):
    '''
    Helper function to create a Bokeh ``Figure`` object
    with a color ramp corresponding to input aggregate and transfer function.

    Parameters
    ----------
    agg : xarray
        Datashader aggregate object (e.g. result of Canvas.points())

    cmap : list of colors or matplotlib.colors.Colormap, optional
        The colormap to use. Can be either a list of colors (in any of the
        formats described above), or a matplotlib colormap object.

    how : str
        Datashader transfer function name (e.g. linear, log, cbrt, eq_hist)

    width : int
        Width in pixels of resulting legend figure (default=600)
    '''

    vals_arr, min_val, max_val = summarize_aggregate_values(agg, how=how)
    img = tf.shade(vals_arr, cmap=cmap, how=how)
    x_axis_type = 'linear' if how == 'linear' else 'log'
    legend_fig = Figure(x_range=(min_val, max_val),
                        plot_height=50,
                        plot_width=width,
                        lod_threshold=None,
                        toolbar_location=None,
                        y_range=(0, 18),
                        x_axis_type=x_axis_type)

    legend_fig.min_border_top = 0
    legend_fig.min_border_bottom = 10
    legend_fig.min_border_left = 15
    legend_fig.min_border_right = 15
    legend_fig.yaxis.visible = False
    legend_fig.grid.grid_line_alpha = 0
    legend_fig.image_rgba(image=[img.values],
                          x=[min_val],
                          y=[0],
                          dw=[max_val - min_val],
                          dh=[18],
                          dw_units='screen')
    return legend_fig

def create_categorical_legend(colormap, aliases=None):
    '''
    Creates a bokeh plot object with circle legend
    swatches and text corresponding to the ``colormap`` key values
    or the optional aliases values.

    Parameters
    ----------
    colormap : dict
        Dictionary of category value to color value

    aliases : dict
        Dictionary of category value to aliases name
    '''
    plot_options = {}
    plot_options['x_range'] = Range1d(start=0, end=200)
    plot_options['y_range'] = Range1d(start=0, end=100)
    plot_options['plot_height'] = 120
    plot_options['plot_width'] = 190
    plot_options['min_border_bottom'] = 0
    plot_options['min_border_left'] = 0
    plot_options['min_border_right'] = 0
    plot_options['min_border_top'] = 0
    plot_options['outline_line_width'] = 0
    plot_options['toolbar_location'] = None

    legend = Plot(**plot_options)

    for i, (cat, color) in enumerate(colormap.items()):
        text_y = 95 - i * 20
        text_val = aliases[cat] if aliases else cat
        legend.add_glyph(Text(x=40,
                              y=text_y-12,
                              text=[text_val],
                              text_font_size='10pt',
                              text_color='#666666'))

        legend.add_glyph(Circle(x=15,
                                y=text_y-5,
                                fill_color=color,
                                size=10,
                                line_color=None,
                                fill_alpha=0.8))

    return legend
