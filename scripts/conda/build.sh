#!/usr/bin/env bash

set -euxo pipefail

git status

export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"
python -m build -w .

git diff --exit-code

VERSION=$(find dist -name "*.whl" -exec basename {} \; | cut -d- -f2)
export VERSION

# Note: pyct is needed in the same environment as conda-build!
conda build scripts/conda/recipe --no-anaconda-upload --no-verify
