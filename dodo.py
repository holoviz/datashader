from pyct import *

# TODO: running the notebooks (nb_tests) not included in all_tests
# because it's too expensive for CI systems.
def task_all_tests():
    return {'actions': None,
            'task_dep': ['unit_tests','lint','nb_lint']}
