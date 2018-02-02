# -*- coding: utf-8 -*-

from nbsite.shared_conf import *

project = u'Datashader'
authors = u'Datashader contributors'
copyright = u'2016-2018, Anaconda, Inc.'
description = 'Turns even the largest data into images, accurately.'

from datashader import __version__
version = release  = __version__

html_static_path += ['_static']
html_theme = 'sphinx_ioam_theme'
html_theme_options = {
    'logo':'datashader-logo.png',
    'favicon':'favicon.ico',
#    'css':'datashader.css'
}

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
        ('Twitter', '//twitter.com/datashader/'),
        ('Github', '//github.com/bokeh/datashader/'),
    )
})

extensions += [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
]

nbbuild_cell_timeout=500
