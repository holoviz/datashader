import os,sys
import shutil
from setuptools import setup

import param
import pyct.build



if __name__ == '__main__':
    example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'datashader','examples')
    if 'develop' not in sys.argv:
        pyct.build.examples(example_path, __file__, force=True)

    version = param.version.get_setup_version(__file__, "datashader", archive_commit="$Format:%h$")

    setup(version=version)

    if os.path.isdir(example_path):
        shutil.rmtree(example_path)
