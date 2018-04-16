import os
from setuptools import find_packages, setup


########## autover ##########

def embed_version(basepath, ref='v0.2.2'):
    """
    Autover is purely a build time dependency in all cases (conda and
    pip) except for when you use pip's remote git support [git+url] as
    1) you need a dynamically changing version and 2) the environment
    starts off clean with zero dependencies installed.
    This function acts as a fallback to make Version available until
    PEP518 is commonly supported by pip to express build dependencies.
    """
    import io, zipfile, importlib
    try:    from urllib.request import urlopen
    except: from urllib import urlopen
    try:
        url = 'https://github.com/ioam/autover/archive/{ref}.zip'
        response = urlopen(url.format(ref=ref))
        zf = zipfile.ZipFile(io.BytesIO(response.read()))
        ref = ref[1:] if ref.startswith('v') else ref
        embed_version = zf.read('autover-{ref}/autover/version.py'.format(ref=ref))
        with open(os.path.join(basepath, 'version.py'), 'wb') as f:
            f.write(embed_version)
        return importlib.import_module("version")
    except:
        return None

def get_setup_version(reponame):
    """
    Helper to get the current version from either git describe or the
    .version file (if available).
    """
    import json
    basepath = os.path.split(__file__)[0]
    version_file_path = os.path.join(basepath, reponame, '.version')
    try:
        from param import version
    except:
        version = embed_version(basepath)
    if version is not None:
        return version.Version.setup_version(basepath, reponame, archive_commit="$Format:%h$")
    else:
        print("WARNING: param>=1.6.0 unavailable. If you are installing a package, this warning can safely be ignored. If you are creating a package or otherwise operating in a git repository, you should install param>=1.6.0.")
        return json.load(open(version_file_path, 'r'))['version_string']



########## dependencies ##########

install_requires = [
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
]

extras_require = {
    # pip doesn't support tests_require
    # (https://github.com/pypa/pip/issues/1197)
    'tests': [
        'pytest >=2.8.5',
        'pytest-benchmark >=3.0.0',
        # TODO: requires numpy headers            
        # 'rasterio',
        'scipy',
        'scikit-image', # was on travis...
        'flake8',
        'nbsmoke >0.2.0',
        'cloudpickle',
        'bokeh'
    ],

    'docs': [
        'nbsite',
    ],
    
    # TODO: Probably needs to be sorted out. Need to remove indirect
    # dependencies unless they must be pinned (or are not actually
    # specified by dependencies)

    # TODO: consider groups of examples, e.g. for tricky dependencies?
    'examples': [
        'attrs',
        'beautifulsoup4',
        'bokeh',
#        'cartopy', # note: you must have already installed numpy & cython to be able to install cartopy, plus ... TODO
        'colorcet',
        # TODO: graphviz on pypi (and requires underling graphviz),
        # python-graphviz for conda (which will correctly pull in
        # graphviz).
        #'graphviz',
        #'python-graphviz',
        'dill',
        'distributed',
        'fastparquet',
        # TODO: no pip package yet
#        'geoviews',
        'holoviews >=1.8.3',
        'ipython', # why? notebook & ipykernel maybe?
        # TODO: see cartopy
#        'iris',
        'jupyter',
        'jupyter_dashboards',
        'krb5',
        'matplotlib',
        'nbconvert',
        'nbformat',
        'networkx >=2.0',
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
}

extras_require['all'] = sorted(set(sum(extras_require.values(), [])))


########## metadata for setuptools ##########

meta = dict(
    name='datashader',
    version=get_setup_version("datashader"),
    description='Data visualization toolchain based on aggregating into a grid',
    url='http://datashader.org',
    maintainer='Datashader developers',
    maintainer_email='dev@datashader.org',
    python_requires=">=2.7",
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=extras_require['tests'],
    license='New BSD',
    packages=find_packages(),
    include_package_data=True
)

if __name__ == '__main__':
    setup(**meta)
