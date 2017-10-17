from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(name='datashader',
      version='0.6.2dev3',
      description='Data visualization toolchain based on aggregating into a grid',
      url='http://github.com/bokeh/datashader',
      install_requires=install_requires,
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True)
