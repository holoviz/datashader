from setuptools import setup
import datashader

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(name='datashader',
      version=datashader.__version__,
      description='Bin based rendering toolchain',
      url='http://github.com/bokeh/datashader',
      install_requires=install_requires,
      packages=['datashader'])
