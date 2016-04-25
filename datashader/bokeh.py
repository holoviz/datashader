import uuid

try:
    import ujson as json
except:
    import json

from bokeh.embed import notebook_div
from bokeh.document import Document
from bokeh.models import CustomJS, ColumnDataSource, Square
from bokeh.model import _ModelInDocument as add_to_document
from bokeh.io import _CommsHandle
from bokeh.util.notebook import get_comms


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
    throttle : int
        The throttle parameter specifies how frequently events
        are generated in milliseconds.
    **kwargs
        Any kwargs provided here will be passed to the callback
        function.
    """

    jscode="""
        // Define a callback to capture errors on the Python side
        function callback(msg){{
            console.log("Python callback returned unexpected message:", msg)
        }}

        function update_plot() {{
            callbacks = {{iopub: {{output: callback}}}};
            var plot = Bokeh.index['{plot_id}'];

            // Generate a command to execute in Python
            var ranges = {{xmin: x_range.attributes.start,
                          ymin: y_range.attributes.start,
                          xmax: x_range.attributes.end,
                          ymax: y_range.attributes.end,
                          w: Math.floor(plot.frame.get('width')),
                          h: Math.floor(plot.frame.get('height'))}}

            var range_str = JSON.stringify(ranges)
            var cmd = "{cmd}(" + range_str + ")"

            // Execute the command on the Python kernel
            if (IPython.notebook.kernel !== undefined) {{
                var kernel = IPython.notebook.kernel;
                kernel.execute(cmd, callbacks, {{silent : false}});
            }}
        }}

        if (!Bokeh._throttle) {{
            Bokeh._throttle = {{}}
        }}

        var throttled_cb = Bokeh._throttle['{ref}'];
        if (throttled_cb) {{
            throttled_cb()
        }} else {{
            Bokeh._throttle['{ref}'] = _.throttle(update_plot, {throttle},
                                                  {{leading: false}});
            Bokeh._throttle['{ref}']()
        }}
    """

    cmd_template = "from {module} import {cls}; {cls}._callbacks['{ref}'].update"

    _callbacks = {}

    def __init__(self, bokeh_plot, callback, throttle=500, **kwargs):
        self.p = bokeh_plot
        self.callback = callback
        self.kwargs = kwargs
        self.ref = str(uuid.uuid4())
        self.comms_handle = None
        self.throttle = throttle

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
        cb_code = cls.jscode.format(plot_id=self.p._id, cmd=cmd,
                                    ref=self.ref.replace('-', '_'),
                                    throttle=self.throttle)
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
            self.comms_handle = _CommsHandle(get_comms(self.ref), self.doc,
                                             {})

        self.update_image(ranges)
        msg = self.get_update_event()
        self.comms_handle.comms.send(msg)


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

    def __init__(self,
                 bokeh_plot,
                 highlight_fill_color='#79DCDE',
                 highlight_line_color='#79DCDE',
                 tooltips=[('Value:', '@value')],
                 size=8):
        """
        Wrapper for adding a HoverTool instance to a plot tools which
        highlights values under the user's mouse location.

        Parameters
        ----------
        bokeh_plot : plot or figure
            Bokeh plot the image will be drawn on

        highlight_fill_color : str
            Fill color for glyph which appears on mouse over.

        highlight_line_color : str
            Line color for glyph which appears on mouse over.

        tooltips : arr
            Tooltip information displayed when user hovers over grid cell
            ex. [('value', '@value')]

        size : int
            Defined hover layer resolution in pixels
            (i.e. height/width of hover grid)
        """

        self.hover_data = ColumnDataSource(data=dict(x=[], y=[], value=[]))

        self.invisible_square = Square(x='x',
                                  y='y',
                                  fill_color=None,
                                  line_color=None,
                                  size=size)

        self.visible_square = Square(x='x',
                                y='y',
                                fill_color=hightlight_fill_color,
                                fill_alpha=.5,
                                line_color=hightlight_line_color,
                                line_alpha=1,
                                size=size)

        code = "source.set('selected', cb_data['index']);"
        self._callback = CustomJS(args={'source': self.hover_data}, code=code)

    def add_to_plot(self, bokeh_plot):

        hover_renderer = bokeh_plot.add_glyph(hover_data,
                                              invisible_square,
                                              selection_glyph=visible_square,
                                              nonselection_glyph=invisible_square)

        self.hover_tool = HoverTool(tooltips=tooltips,
                                    callback=self._callback,
                                    renderers=[hover_renderer],
                                    mode='mouse')

        bokeh_plot.add_tools(self.hover_tool)
