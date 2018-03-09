import os
import importlib
import json
from setuptools import find_packages, setup

def embed_version(basepath, ref='v0.2.1'):
    """
    Autover is purely a build time dependency in all cases (conda and
    pip) except for when you use pip's remote git support [git+url] as
    1) you need a dynamically changing version and 2) the environment
    starts off clean with zero dependencies installed.

    This function acts as a fallback to make Version available until
    PEP518 is commonly supported by pip to express build dependencies.
    """
    import io, zipfile
    try:    from urllib.request import urlopen
    except: from urllib import urlopen
    response = urlopen('https://github.com/ioam/autover/archive/{ref}.zip'.format(ref=ref))
    zf = zipfile.ZipFile(io.BytesIO(response.read()))
    ref = ref[1:] if ref.startswith('v') else ref
    embed_version = zf.read('autover-{ref}/autover/version.py'.format(ref=ref))
    with open(os.path.join(basepath, 'version.py'), 'wb') as f:
        f.write(embed_version)


def get_setup_version(reponame):
    """
    Helper to get the current version from either git describe or the
    .version file (if available).
    """
    basepath = os.path.split(__file__)[0]
    version_file_path = os.path.join(basepath, reponame, '.version')
    version = None
    try: version = importlib.import_module(reponame + ".version") # Bundled
    except:  # autover available as package
        try: from autover import version
        except:
            try: from param import version # Try to get it from param
            except:
                embed_version(basepath)
                version = importlib.import_module("version")

    # TODO: try/except below until new param release
    try:
        if version is not None:
            return version.Version.setup_version(basepath, reponame, archive_commit="$Format:%h$")
        else:
            print("WARNING: autover unavailable. If you are installing a package, this warning can safely be ignored. If you are creating a package or otherwise operating in a git repository, you should refer to autover's documentation to bundle autover or add it as a dependency.")
            return json.load(open(version_file_path, 'r'))['version_string']
    except:
        return '0.0.0+unknown'

meta = dict(
    name='datashader',
    version=get_setup_version("datashader"),
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
    extras_require={
        # tests_require not supported by pip
        'tests': [
            'pytest >=2.8.5',
            'pytest-benchmark >=3.0.0',
# TODO: requires numpy headers            
#            'rasterio',
            'scipy',
            'scikit-image', # was on travis...
            'flake8',
            'nbsmoke',
            'cloudpickle',
            'bokeh'
        ],
        # TODO: need to remove indirect dependencies unless they must be
        # pinned (or are not actually specified by dependencies)
        'examples': [
            'attrs',
            'beautifulsoup4',
            'bokeh',
            'cartopy',
            'colorcet',
            'graphviz',
            'pytest',
            'pytest-benchmark',
            'python-graphviz',
            'dask >=0.15.4',
            'datashader',
            'dill',
            'distributed',
            'fastparquet',
            'geoviews',
            'holoviews >=1.8.3',
            'ipython', # why? notebook & ipykernel maybe?
            'iris',
            'jupyter',
            'jupyter_dashboards',
            'krb5',
            'matplotlib',
            'nbconvert',
            'nbformat',
            'networkx >=2.0',
            'numba',
            'numpy',
            'pandas',
            'paramnb',
            'pyproj',
            'pytables',
            'python-snappy',
            'rasterio',
            'requests',
            'scipy',
            'shapely',
            'snappy',
            'statsmodels',
            'tblib',
            'xarray',
            'yaml',
# TODO: needs conda package?            
#            'cachey',
            'streamz ==0.2.0',
            'webargs'
        ]
    },
    license='New BSD',
    packages=find_packages(),
    package_data={'datashader': ['.version']},
    include_package_data=True
)

# TODO: decide later what to do about this (share on win?)
# Copy all examples files but exclude IPython checkpoints
#cp -r $SRC_DIR/examples $PREFIX/share/datashader-examples
#rm -rf $PREFIX/share/datashader-examples/.ipynb_checkpoints

if __name__ == '__main__':
    setup(**meta)
