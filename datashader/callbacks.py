import uuid, json

from bokeh.embed import notebook_div
from bokeh.document import Document
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.model import _ModelInDocument as add_to_document
from bokeh.io import _CommsHandle
from bokeh.util.notebook import get_comms


class InteractiveImage(object):
    """
    Bokeh-based interactive image object that updates on pan/zoom events.

    Given a Bokeh plot and a callback function, calls the function
    whenever the pan or zoom changes the plot's extent, regenerating
    the image dynamically.  Works in a Jupyter/IPython notebook cell,
    using the existing notebook kernel Python process (not a separate
    Bokeh server).  Does not yet support usage outside the notebook,
    but could be extened to use Bokeh server in that case.  
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
                          w: plot.frame.get('width'),
                          h: plot.frame.get('height')}}

            var range_str = JSON.stringify(ranges)
            var cmd = "{cmd}(" + range_str + ")"

            // Execute the command on the Python kernel
            var kernel = IPython.notebook.kernel;
            kernel.execute(cmd, callbacks, {{silent : false}});
        }}

        if (!Bokeh._throttle) {{
            Bokeh._throttle = {{}}
        }}

        var throttled_cb = Bokeh._throttle['{ref}'];
        if (throttled_cb) {{
            throttled_cb()
        }} else {{
            Bokeh._throttle['{ref}'] = _.debounce(update_plot, {throttle});
            Bokeh._throttle['{ref}']()
        }}
    """

    cmd_template = "from {module} import {cls}; {cls}._callbacks['{ref}'].update"

    _callbacks = {}

    def __init__(self, bokeh_plot, callback, throttle=100, **kwargs):
        """
        The callback function should have the signature:

        fn(x_range=(xmin, xmax), y_range=(ymin, ymax), w, h, **kwargs)

        and return a PIL image object.  Any kwargs provided here will
        be passed to the callback each time.

        The throttle parameter allows control over how many times the
        callback will get executed when there are frequent closely
        spaced events.        
        """

        self.p = bokeh_plot
        self.callback = callback
        self.kwargs = kwargs

        # Initialize RGBA image glyph and datasource
        w, h = self.p.plot_width, self.p.plot_height
        xmin, xmax = self.p.x_range.start, self.p.x_range.end
        ymin, ymax = self.p.y_range.start, self.p.y_range.end
        dw, dh = xmax-xmin, ymax-ymin
        image = self.callback(x_range=(xmin, xmax), y_range=(ymin, ymax), w=w, h=h, **self.kwargs)

        self.ds = ColumnDataSource(data=dict(image=[image.img], x=[xmin],
                                             y=[ymin], dw=[dw], dh=[dh]))
        self.p.image_rgba(source=self.ds, image='image', x='x', y='y',
                          dw='dw', dh='dh', dilate=False)

        # Register callback on the class with unique reference
        cls = type(self)
        self.ref = str(uuid.uuid4())
        cls._callbacks[self.ref] = self

        # Generate python callback command
        cmd = cls.cmd_template.format(module=cls.__module__,
                                      cls=cls.__name__, ref=self.ref)

        # Initialize callback
        cb_code = cls.jscode.format(plot_id=self.p._id, cmd=cmd,
                                    ref=self.ref.replace('-', '_'),
                                    throttle=throttle)
        cb_args = dict(x_range=self.p.x_range, y_range=self.p.y_range)
        callback = CustomJS(args=cb_args, code=cb_code)
        self.p.x_range.callback = callback
        self.p.y_range.callback = callback

        # Initialize document
        doc_handler = add_to_document(self.p)
        with doc_handler:
            self.doc = doc_handler._doc
            self.div = notebook_div(self.p, self.ref)
        self.comms = None


    def update(self, ranges):
        if not self.comms:
            self.comms = _CommsHandle(get_comms(self.ref), self.doc,
                                      self.doc.to_json())

        self.redraw_image(ranges)
        to_json = self.doc.to_json()
        msg = Document._compute_patch_between_json(self.comms.json, to_json)
        self.comms._json[self.doc] = to_json
        self.comms.comms.send(json.dumps(msg))


    def redraw_image(self, ranges):
        x_range = (ranges['xmin'], ranges['xmax'])
        y_range = (ranges['ymin'], ranges['ymax'])
        dh = y_range[1] - y_range[0]
        dw = x_range[1] - x_range[0]

        image = self.callback(x_range=x_range, y_range=y_range, w=ranges['w'], h=ranges['h'], **self.kwargs)
        self.ds.data.update(dict(image=[image.img], x=[x_range[0]], y=[y_range[0]],
                                 dw=[dw], dh=[dh]))


    def _repr_html_(self):
        return self.div


IPythonKernelCallback = InteractiveImage  # Temporary; for backwards compatibility
