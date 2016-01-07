from __future__ import absolute_import, division, print_function

from datashape.dispatch import namespace
from multipledispatch import dispatch
from functools import partial


__all__ = 'dispatch', 'namespace'


dispatch = partial(dispatch, namespace=namespace)
