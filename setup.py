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
    'datashape',
    'numba >=0.51',
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

examples = [
    'bokeh <3.0',
    'geopandas',
    'holoviews',
    'matplotlib',
    'scikit-image',
    'spatialpandas',
]

extras_require = {
    'tests': [
        'codecov',
        'fastparquet',  # optional dependency
        'flake8',
        'nbconvert',
        'nbformat <=5.4.0',
        'nbsmoke[verify] >0.5',
        'netcdf4',
        'pyarrow',
        'pytest',
        'pytest-benchmark',
        'pytest-cov',
        'rasterio',
        'rioxarray',
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
        'snappy',
    ]
}

extras_require['doc'] = extras_require['examples_extra'] + [
    'nbsite >=0.7.1',
    'numpydoc',
    'pydata-sphinx-theme <0.9.0',
    'sphinx-copybutton',
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
    python_requires=">=3.7",
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
