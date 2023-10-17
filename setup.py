import os,sys
import shutil
from setuptools import find_packages, setup

# build dependencies
import param
import pyct.build

########## dependencies ##########

install_requires = [
    'colorcet',
    'dask',
    'multipledispatch',
    'numba',
    'numpy',
    'pandas',
    'param',
    'pillow',
    'pyct',
    'requests',
    'scipy',
    'toolz',
    'xarray',
]

geopandas = [
    'dask-geopandas',
    'geopandas',
    'shapely >=2.0.0',
]

examples = geopandas + [
    'bokeh >3.1',
    'geodatasets',
    'holoviews',
    'matplotlib >=3.3',
    'panel >1.1',
    'scikit-image',
    'spatialpandas',
]

extras_require = {
    'tests': geopandas + [
        'codecov',
        'geodatasets',
        'fastparquet',  # optional dependency
        'flake8',
        'nbconvert',
        'nbformat',
        'nbsmoke[verify] >0.5',
        'netcdf4',
        'pyarrow',
        'pytest',
        'pytest-benchmark',
        'pytest-cov',
        'rasterio',
        'rioxarray',
        'scikit-image',
        'spatialpandas',
    ],
    'examples': examples,
    'examples_extra': examples + [
        'networkx',
        'streamz',
        ### conda only below here
        'fastparquet',
        'graphviz',
        'python-graphviz',
        'python-snappy',
        'rasterio',
    ],
    'geopandas': geopandas,
}


extras_require['doc'] = extras_require['examples_extra'] + [
    'nbsite >=0.8.2,<0.9.0',
    'numpydoc',
]

extras_require['all'] = sorted(set(sum(extras_require.values(), [])))

extras_require['gpu_tests'] = [
    "cupy",
    "cudf",  # Install with conda install -c rapidsai
    "dask-cudf",  # Install with conda install -c rapidsai
]

########## metadata for setuptools ##########

setup_args = dict(
    name='datashader',
    version=param.version.get_setup_version(__file__,"datashader",archive_commit="$Format:%h$"),
    description='Data visualization toolchain based on aggregating into a grid',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url='https://datashader.org',
    project_urls={
        'Source': 'https://github.com/holoviz/datashader',
    },
    maintainer='Datashader developers',
    maintainer_email='dev@datashader.org',
    python_requires=">=3.9",
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
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries",
    ]
)

if __name__ == '__main__':
    example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'datashader','examples')
    if 'develop' not in sys.argv:
        pyct.build.examples(example_path, __file__, force=True)

    setup(**setup_args)

    if os.path.isdir(example_path):
        shutil.rmtree(example_path)
