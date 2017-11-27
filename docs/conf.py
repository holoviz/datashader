# -*- coding: utf-8 -*-

from nbsite.shared_conf import *

project = u'datashader'
authors = u'datashader contributors'
copyright = u'2016, Continuum Analytics'
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
    ('Performance', 'performance'),
    ('API', 'api'),
    ('FAQ', 'FAQ')
)

html_context.update({
    'PROJECT': project,
    'DESCRIPTION': description,
    'AUTHOR': authors,
    'WEBSITE_SERVER': 'https://bokeh.github.io/datashader',
    'VERSION': version,
    'NAV': _NAV,
    'LINKS': _NAV,
    'SOCIAL': (
        ('Twitter', '//twitter.com/datashader/'),
        ('Github', '//github.com/bokeh/datashader/'),
    )
})
