from nbsite.shared_conf import *

project = 'Datashader'
copyright_years['start_year'] = '2016'
copyright = copyright_fmt.format(**copyright_years)
description = 'Turns even the largest data into images, accurately.'

from datashader import __version__
version = release  = base_version(__version__)

html_static_path += ['_static']
html_theme = 'pydata_sphinx_theme'

html_css_files += [
    'css/custom.css'
]

# Somehow .ipynb started to take precedence over .rst.
# This broke the landing page `index.rst` as it embeds `index.ipynb` file.
# Adding `.md` to make our life easier in the future.
source_suffix = ['.rst', '.md', '.ipynb']
master_doc = 'index'

html_logo = '_static/logo_horizontal.svg'
html_favicon = '_static/favicon.ico'
html_show_sourcelink = False

html_theme_options.update({
    'github_url': 'https://github.com/holoviz/datashader',
    'icon_links': [
        {
            'name': 'X',
            'url': 'https://x.com/datashader',
            'icon': 'fa-brands fa-square-x-twitter',
        },
        {
            'name': 'Discourse',
            'url': 'https://discourse.holoviz.org/c/datashader/',
            'icon': 'fa-brands fa-discourse',
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/AXRHnJU6sP",
            "icon": "fa-brands fa-discord",
        },
    ],
    "pygments_dark_style": "material"
})

html_context.update({
    # Used to add binder links to the latest released tag.
    'last_release': f'v{release}',
    'github_user': 'holoviz',
    'github_repo': 'datashader',
})

extensions += [
    'sphinx.ext.autosummary',
    'numpydoc',
    'nbsite.analytics',
    'sphinxcontrib.mermaid',
    'sphinx_reredirects',
]

myst_fence_as_directive = ["mermaid"]

nbsite_analytics = {
    'goatcounter_holoviz': True,
}

nbbuild_cell_timeout = 2000

redirects = {
    'topics/index': 'https://examples.holoviz.org',
}

autosummary_generate = True

# Datashader uses sphinx.ext.autodoc (e.g. automodule) for its API reference
# and automatically include a module that contains Image. Image inherits
# from xr.DataArray. Datashader uses numpydoc to parse the docstrings.
# It turns out xarray broke numpydoc https://github.com/pydata/xarray/issues/8596
# This is a bad hack to work around this issue.

import numpydoc.docscrape  # noqa

original_error_location = numpydoc.docscrape.NumpyDocString._error_location

def patch_error_location(self, msg, error=True):
    try:
        original_error_location(self, msg, error)
    except ValueError as e:
        if "site-packages/xarray" in str(e):
            return
        else:
            raise e

numpydoc.docscrape.NumpyDocString._error_location = patch_error_location

# Override the Sphinx default title that appends `documentation`
html_title = f'{project} v{version}'

# /Users/runner/work/datashader/datashader/datashader/core.py:docstring of datashader.core.Canvas:21:
# WARNING: autosummary: stub file not found 'datashader.Canvas.area'. Check your autosummary_generate setting.
# See https://stackoverflow.com/a/73294408
numpydoc_class_members_toctree = False

numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

exclude_patterns = ['governance']
