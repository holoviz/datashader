To build a local copy of the datashader docs, run these commands::

  git clone git@github.com:bokeh/datashader.git

  cd datashader
  conda create -n datashader-docs python
  source activate datashader-docs
  conda install -c bokeh --file requirements.txt
  python setup.py develop

  cd docs
  conda install --file requirements-docs.txt
  make html
  open build/html/index.html
