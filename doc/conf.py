# -*- coding: utf-8 -*-

from nbsite.shared_conf import *

project = u'Datashader'
authors = u'Datashader contributors'
copyright = u'2016-2019, Anaconda, Inc.'
description = 'Turns even the largest data into images, accurately.'

from datashader import __version__
version = release  = __version__

html_static_path += ['_static']
html_theme = 'sphinx_ioam_theme'
html_theme_options = {
    'logo':'datashader-logo.png',
    'favicon':'favicon.ico',
    'css':'site.css'
}

templates_path = ['_templates']

_NAV =  (
    ('Getting Started', 'getting_started/index'),
    ('User Guide', 'user_guide/index'),
    ('Topics', 'topics/index'),
    ('API', 'api'),
    ('FAQ', 'FAQ')
)

html_context.update({
    'PROJECT': project,
    'DESCRIPTION': description,
    'AUTHOR': authors,
    'WEBSITE_SERVER': 'http://datashader.org',
    'VERSION': version,
    'NAV': _NAV,
    'LINKS': _NAV,
    'SOCIAL': (
        ('Github', 'https://github.com/bokeh/datashader/'),
        ('Twitter', 'https://twitter.com/datashader/'),
        ('Gitter', 'https://gitter.im/pyviz/pyviz'),
        ('PyViz', 'http://pyviz.org'),
    )
})

extensions += [
    'sphinx.ext.autosummary',
    'numpydoc',
]

nbbuild_cell_timeout=2000
