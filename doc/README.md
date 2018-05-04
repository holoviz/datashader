See https://pyviz.github.io/nbsite/Usage.html for more details.

First, set up environment so you can run the examples and build docs:

```
$ conda install -c pyviz/label/dev pyctdev # if you don't already have pyctdev
$ doit env_create -c pyviz/label/dev -c conda-forge --python=3.6 --name=dsdocs
$ conda activate dsdocs
$ doit develop_install -c pyviz/label/dev -c conda-forge -o doc
$ datashader fetch-data --path=examples
```

(optional) Building the docs does not check the notebooks run without errors (you have to watch out for tracebacks flying by). Building the docs also runs the notebooks with modifications (e.g. setting backend options). If you want to be sure all the notebooks run normally without exception, execute `doit test_examples_extra`. (Requires running the notebooks twice; this is future work for nbsite.)

Build the docs (note: it's future pyctdev/nbsite work to make this simpler):

1. `cd doc`

2. Generate rst containers for notebooks:
   `nbsite_nbpagebuild.py bokeh datashader ../examples .`

3. Build site: `sphinx-build -b html . ./_build/html` followed by
   `nbsite_fix_links.py _build/html`

4. Inspect result: `pushd _build/html && python -m http.server && popd`

5. Clean up for deployment: `nbsite_cleandisthtml.py ./_build/html take_a_chance`

6. Deploy to S3 bucket: `pushd _build/html && aws s3 sync --delete --acl public-read . s3://datashader.org && popd`
