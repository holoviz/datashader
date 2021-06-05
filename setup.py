import os,sys
import shutil
from setuptools import find_packages, setup

# build dependencies
import param
import pyct.build

########## dependencies ##########

install_requires = [
    # `conda install dask[complete]` happily gives you dask...which is
    # happily like pip's dask[complete]. (conda's dask-core is more
    # like pip's dask.)
    'dask[complete] >=0.18.0',
    'datashape >=0.5.1',
    'numba >=0.51',
    'numpy >=1.7',
    'pandas >=0.24.1',
    'pillow >=3.1.1',
    'xarray >=0.9.6',
    'colorcet >=0.9.0',
    'param >=1.6.1',
    'pyct >=0.4.5',
    'scipy',
]

examples = [
    'holoviews >=1.10.0',
    'scikit-image',
    'bokeh',
    'matplotlib',
]

extras_require = {
    'tests': [
        'pytest >=3.9.3,<6.0',
        'pytest-benchmark >=3.0.0',
        'pytest-cov',
        'codecov',
        'flake8',
        'nbconvert <6',
        'nbsmoke[all] >=0.4.0',
        'fastparquet >=0.1.6',  # optional dependency
        'holoviews >=1.10.0',
        'bokeh',
        'pyarrow',
        'netcdf4',
        'twine',   # required for pip packaging
        'rfc3986', # required by twine
        'keyring', # required by twine
        'spatialpandas'
    ],
    'examples': examples,
    'examples_extra': examples + [
        'networkx >=2.0',
        'streamz >=0.2.0',
        ### conda only below here
        'graphviz',
        'python-graphviz',
        'fastparquet',
        'python-snappy',
        'rasterio',
        'snappy',
    ]
}

extras_require['doc'] = extras_require['examples_extra'] + [
    'nbsite >=0.5.2',
    'sphinx_holoviz_theme',
    'numpydoc'
]

extras_require['all'] = sorted(set(sum(extras_require.values(), [])))



########## metadata for setuptools ##########

setup_args = dict(
    name='datashader',
    version=param.version.get_setup_version(__file__,"datashader",archive_commit="$Format:%h$"),
    description='Data visualization toolchain based on aggregating into a grid',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url='https://datashader.org',
    maintainer='Datashader developers',
    maintainer_email='dev@datashader.org',
    python_requires=">=2.7",
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=extras_require['tests'],
    license='New BSD',
    packages=find_packages(),
    include_package_data = True,
    entry_points={
        'console_scripts': [
            'datashader = datashader.__main__:main'
        ]
    },
)

if __name__ == '__main__':
    example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'datashader','examples')
    if 'develop' not in sys.argv:
        pyct.build.examples(example_path, __file__, force=True)

    setup(**setup_args)

    if os.path.isdir(example_path):
        shutil.rmtree(example_path)
