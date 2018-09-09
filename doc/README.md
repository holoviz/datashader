See https://pyviz.github.io/nbsite/Usage.html for more details.

First, set up environment so you can run the examples and build docs:

```
$ conda install -c pyviz/label/dev pyctdev # if you don't already have pyctdev
$ doit env_create -c pyviz/label/dev -c conda-forge --python=3.6 --name=dsdocs
$ conda activate dsdocs
$ doit develop_install -c pyviz/label/dev -c defaults -c conda-forge -o doc
$ datashader fetch-data --path=examples
```

WARNING: when you run `develop_install` above, which uses python/pip
to do a develop install, if you have data in your examples/data
directory, it will be copied around at least one time :( So
temporarily move it away before doing a develop install, then restore
it afterwards. (This applies to doing a pip editable install any time,
not just for the documentation; see
https://github.com/pyviz/pyct/issues/22)

(optional) Building the docs does not check the notebooks run without errors (you have to watch out for tracebacks flying by). Building the docs also runs the notebooks with modifications (e.g. setting backend options). If you want to be sure all the notebooks run normally without exception, execute `doit test_examples_extra`. (Requires running the notebooks twice; this is future work for nbsite.)

Build the docs (note: it's future pyctdev/nbsite work to make this simpler):

1. Generate rst containers for notebooks:
   `nbsite generate-rst --org pyviz --project-name datashader --repo datashader`

2. Build site:
   `nbsite build --what=html --output=builtdocs`

3. Inspect result: `pushd builtdocs && python -m http.server && popd`

4. Edit notebooks as desired and repeat steps 1-3 as required. Unedited notebooks will not be re-run.

5. Clean up for deployment: `nbsite_cleandisthtml.py builtdocs take_a_chance`

6. Deploy to S3 bucket: `pushd builtdocs && aws s3 sync --delete --acl public-read . s3://datashader.org && popd`
