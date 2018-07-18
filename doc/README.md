See https://pyviz.github.io/nbsite/Usage.html for more details.

First, set up environment so you can run the examples and build docs:

```
$ conda install -c pyviz/label/dev pyctdev # if you don't already have pyctdev
$ doit env_create -c pyviz/label/dev -c conda-forge --python=3.6 --name=dsdocs
$ conda activate dsdocs
$ doit develop_install -c pyviz/label/dev -c defaults -c conda-forge -o doc
$ datashader fetch-data --path=examples
```

(optional) Building the docs does not check the notebooks run without errors (you have to watch out for tracebacks flying by). Building the docs also runs the notebooks with modifications (e.g. setting backend options). If you want to be sure all the notebooks run normally without exception, execute `doit test_examples_extra`. (Requires running the notebooks twice; this is future work for nbsite.)

Build the docs (note: it's future pyctdev/nbsite work to make this simpler):

1. Generate rst containers for notebooks:
   `nbsite generate-rst --org bokeh --project datashader --repo datashader --examples-path examples --doc-path doc`

2. Build site:
   `nbsite build --what=html --examples-path=examples --doc-path=doc --output=builtdocs`

3. Inspect result: `pushd builtdocs && python -m http.server && popd`

4. Clean up for deployment: `nbsite_cleandisthtml.py builtdocs take_a_chance`

5. Deploy to S3 bucket: `pushd builtdocs && aws s3 sync --delete --acl public-read . s3://datashader.org && popd`
