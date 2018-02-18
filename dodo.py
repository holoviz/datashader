from pyct import *

def task_all_tests():
    return {'actions': [],
            'task_dep': ['nb_tests','nb_lint','unit_tests','lint']}
