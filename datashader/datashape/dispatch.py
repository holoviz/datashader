from multipledispatch import dispatch
from functools import partial

namespace = {}

dispatch = partial(dispatch, namespace=namespace)
