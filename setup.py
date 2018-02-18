import os
from setuptools import find_packages, setup

meta = dict(
    name='datashader',
    version=os.environ.get("VERSIONHACK","0.6.5"),
    description='Data visualization toolchain based on aggregating into a grid',
    url='http://datashader.org',
    python_requires=">=2.7",
    install_requires=[
        'dask >=0.15.4',
        'datashape >=0.5.1',
        'numba >=0.35.0',
        'numpy >=1.7',
        'pandas >=0.20.3',
        'pillow >=3.1.1',
        'xarray >=0.9.6',
        'toolz >=0.7.4',
        'colorcet >=0.9.0',
        'param >=1.5.0,<2.0',
    ],
    tests_require=[
        'pytest >=2.8.5',
        'pytest-benchmark >=3.0.0',
        'rasterio',
        'scipy',
        'scikit-image', # was on travis...
        'flake8', # was on travis
        'nbsmoke'
    ],
    license='New BSD',
    packages=find_packages(),
    include_package_data=True
)

# TODO: decide later what to do about this (share on win?)
# Copy all examples files but exclude IPython checkpoints
#cp -r $SRC_DIR/examples $PREFIX/share/datashader-examples
#rm -rf $PREFIX/share/datashader-examples/.ipynb_checkpoints

if __name__ == '__main__':
    setup(**meta)
