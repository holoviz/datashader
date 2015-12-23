import sys

import numba as nb


ngjit = nb.jit(nopython=True, nogil=True)


if sys.version_info.major == 3:
    def __exec(codestr, glbls):
        exec(codestr, glbls)
else:
    eval(compile("""
def __exec(codestr, glbls):
    exec codestr in glbls
""",
                 "<_exec>", "exec"))


def _exec(code, namespace, debug=False):
    if debug:
        print(("Code:\n-----\n{0}\n"
              "Namespace:\n----------\n{1}").format(code, namespace))
    __exec(code, namespace)
