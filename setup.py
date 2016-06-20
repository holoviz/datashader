from setuptools import setup

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(name='datashader',
      version='0.2.0',
      description='Data visualization toolchain based on aggregating into a grid',
      url='http://github.com/bokeh/datashader',
      install_requires=install_requires,
      packages=['datashader', 'datashader.tests'])
