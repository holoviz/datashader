# -*- coding: utf-8 -*-

from nbsite.shared_conf import *

project = u'Datashader'
authors = u'Datashader contributors'
copyright = u'2016-2022, Anaconda, Inc.'
description = 'Turns even the largest data into images, accurately.'

from datashader import __version__
version = release  = base_version(__version__)

html_static_path += ['_static']
html_theme = 'pydata_sphinx_theme'

html_css_files = [
    'nbsite.css',
    'css/custom.css'
]

html_logo = '_static/logo_horizontal.svg'
html_favicon = '_static/favicon.ico'

html_theme_options = {
    'github_url': 'https://github.com/holoviz/datashader',
    'icon_links': [
        {
            'name': 'Twitter',
            'url': 'https://twitter.com/datashader',
            'icon': 'fab fa-twitter-square',
        },
        {
            'name': 'Discourse',
            'url': 'https://discourse.holoviz.org/c/datashader/',
            'icon': 'fab fa-discourse',
        },
    ],
    "footer_items": [
        "copyright",
        "last-updated",
    ],
    'google_analytics_id': 'UA-154795830-1',
}

templates_path = [
    '_templates'
]

html_context.update({
    # Used to add binder links to the latest released tag.
    'last_release': f'v{release}',
    'github_user': 'holoviz',
    'github_repo': 'datashader',
})


extensions += [
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx_copybutton',
]

nbbuild_cell_timeout = 2000

# Override the Sphinx default title that appends `documentation`
html_title = f'{project} v{version}'
# Format of the last updated section in the footer
html_last_updated_fmt = '%Y-%m-%d'
