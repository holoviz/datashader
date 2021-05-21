# -*- coding: utf-8 -*-

from nbsite.shared_conf import *

project = u'Datashader'
authors = u'Datashader contributors'
copyright = u'2016-2021, Anaconda, Inc.'
description = 'Turns even the largest data into images, accurately.'

from datashader import __version__
version = release  = __version__

html_static_path += ['_static']
html_theme = 'sphinx_holoviz_theme'
html_theme_options = {
    'logo': 'logo_horizontal.svg',
    'include_logo_text': False,
    'favicon': 'favicon.ico',
    'primary_color': '#774c9a',
    'primary_color_dark': '#460f72',
    'secondary_color': '#c4a9d6',
    'second_nav': True,
}

_NAV =  (
    ('Getting Started', 'getting_started/index'),
    ('User Guide', 'user_guide/index'),
    ('Topics', 'topics/index'),
    ('API', 'api'),
    ('FAQ', 'FAQ'),
    ('About', 'about')
)

html_context.update({
    'PROJECT': project,
    'DESCRIPTION': description,
    'AUTHOR': authors,
    'WEBSITE_SERVER': 'https://datashader.org',
    'GOOGLE_SEARCH_ID': '017396756996884923145:fgzzciei5qk',
    'GOOGLE_ANALYTICS_UA': 'UA-154795830-1',
    'VERSION': version,
    'NAV': _NAV,
    'LINKS': _NAV,
    'SOCIAL': (
        ('Github', 'https://github.com/holoviz/datashader/'),
        ('Twitter', 'https://twitter.com/datashader/'),
        ('Discourse', 'https://discourse.holoviz.org/'),
        ('HoloViz', 'https://holoviz.org'),
    )
})

extensions += [
    'sphinx.ext.autosummary',
    'numpydoc',
]

nbbuild_cell_timeout=2000
