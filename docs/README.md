See https://ioam.github.io/nbsite/Usage.html for more details.

0. Set up environment so you can run the examples

1. Install nbsite: `conda install -c conda-forge sphinx beautifulsoup4
   graphviz && pip install sphinx_ioam_theme nbsite`

2. `cd docs`

3. Generate rst containers for notebooks: `nbsite_nbpagebuild.py bokeh
   datashader-docs ../examples .`

4. Build site: `sphinx-build -b html . ./_build/html` followed by
   `nbsite_fix_links.py _build/html`

5. Inspect result: `pushd _build/html && python -m http.server`

6. Clean up for deployment: `nbsite_cleandisthtml.py ./_build/html take_a_chance`

7. Deploy: `cd /path/to/bokeh/datashader-docs && git checkout gh-pages
   && git rm -rf . && cp -r
   /path/to/bokeh/datashader/docs/_build/html/* . && git add . && git
   commit -m "New site..." && git push`. Note: will cause
   datashader-docs repo to grow - might be better to hard git reset
   and push force to just write over current contents.

