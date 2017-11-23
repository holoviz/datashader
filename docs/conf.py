# -*- coding: utf-8 -*-


from nbsite.shared_conf import * # noqa

##############################################################
# start of things to edit

project = u'datashader'
copyright = u'2016, Continuum Analytics'
authors = u'datashader contributors'

# TODO: rename
ioam_module = 'holoviews'
description = 'Stop plotting your data - annotate your data and let it visualize itself.'

# TODO: gah, version
from datashader import __version__ as version
version = version
release = version

html_static_path = ['_static']

html_theme = 'sphinx_ioam_theme'
html_theme_options = {
#    'logo':'images/amazinglogo.png'
#    'favicon':'images/amazingfavicon.ico'
# ...
# ? css
# ? js
}


_NAV =  (
        ('Getting Started', 'getting_started/index'),
        ('User Guide', 'user_guide/index'),
        ('Topics', 'topics/index'),
        ('Performance', 'performance'),
        ('API', 'api'),
        ('FAQ', 'FAQ'))

html_context = {
    'PROJECT': project,
    'DESCRIPTION': description,
    'AUTHOR': authors,
    # will work without this - for canonical (so can ignore when building locally or test deploying)    
    'WEBSITE_SERVER': 'https://bokeh.github.io/datashader',
    'VERSION': version,
    'NAV': _NAV,
    'LINKS': _NAV,
    'SOCIAL': (
#        ('Gitter', '//gitter.im/ioam/holoviews'),
        ('Twitter', '//twitter.com/datashader/'),
        ('Github', '//github.com/bokeh/datashader/'),
    ),
    'js_includes': ['custom.js', 'require.js'],
}

# end of things to edit
##############################################################

from nbsite.shared_conf2 import hack
setup, intersphinx_mapping, texinfo_documents, man_pages, latex_documents, htmlhelp_basename, html_static_path, html_title, exclude_patterns = hack(project,ioam_module,authors,description,html_static_path)
