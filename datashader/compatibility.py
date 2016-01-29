from __future__ import absolute_import, division, print_function

import sys

PY3 = sys.version_info[0] == 3

if PY3:
    def apply(func, args, kwargs=None):
        if kwargs:
            return func(*args, **kwargs)
        else:
            return func(*args)
else:
    apply = apply
