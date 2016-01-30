import uuid

from bokeh.models import CustomJS
from bokeh.io import push_notebook


class IPythonKernelCallback(object):
    
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
            console.log(throttled_cb)
        }} else {{
            Bokeh._throttle['{ref}'] = _.throttle(update_plot, {throttle});
            console.log(Bokeh._throttle['{ref}'])
            Bokeh._throttle['{ref}']()
        }}
    """
    
    cmd_template = "from {module} import {cls}; {cls}._callbacks['{ref}'].update_image"
    
    _callbacks = {}

    def __init__(self, bokeh_plot, callback, throttle=250, **kwargs):
        self.p = bokeh_plot
        self.callback = callback
        self.kwargs = kwargs

        # Register callback on the class with unique reference
        cls = type(self)
        ref = str(uuid.uuid4())
        cls._callbacks[ref] = self

        # Generate python callback command
        cmd = cls.cmd_template.format(module=cls.__module__,
                                      cls=cls.__name__, ref=ref)

        # Initialize callback
        cb_code = cls.jscode.format(plot_id=self.p._id, cmd=cmd,
                                    ref=ref.replace('-', '_'),
                                    throttle=throttle)
        cb_args = dict(x_range=self.p.x_range, y_range=self.p.y_range)
        callback = CustomJS(args=cb_args, code=cb_code)
        self.p.x_range.callback = callback
        self.p.y_range.callback = callback

    def update_image(self, ranges):
        self.p.renderers.pop()
        ranges['x_range'] = (ranges['xmin'], ranges['xmax'])
        ranges['y_range'] = (ranges['ymin'], ranges['ymax'])
        self.callback(ranges, **self.kwargs)
        push_notebook(document=self.p.document)
