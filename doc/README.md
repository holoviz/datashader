See https://pyviz.github.io/nbsite/Usage.html for more details.

0. Set up environment so you can run the examples and build docs (e.g. `conda install -c pyviz/label/dev pyct` then `doit env_create -c pyviz/label/dev -c conda-forge --python=3.6 --name=dsdocs` then `conda activate dsdocs` then `doit develop_install -c pyviz/label/dev -c conda-forge -o doc`).

2. `cd doc`

3. Generate rst containers for notebooks:
   `nbsite_nbpagebuild.py bokeh datashader ../examples .`

4. Build site: `sphinx-build -b html . ./_build/html` followed by
   `nbsite_fix_links.py _build/html`

5. Inspect result: `pushd _build/html && python -m http.server`

6. Clean up for deployment: `nbsite_cleandisthtml.py ./_build/html take_a_chance`

7. Deploy to S3 bucket: `pushd _build/html && aws s3 sync --delete --acl public-read . s3://datashader.org`
