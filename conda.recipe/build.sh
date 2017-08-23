#!/bin/bash

set -e
set -x

BLD_DIR=`pwd`

SRC_DIR=$RECIPE_DIR/..
pushd $SRC_DIR

$PYTHON setup.py --quiet install --single-version-externally-managed --record=record.txt

# Copy all examples files but exclude IPython checkpoints
cp -r $SRC_DIR/examples $PREFIX/share/datashader-examples
rm -rf $PREFIX/share/datashader-examples/.ipynb_checkpoints

popd

