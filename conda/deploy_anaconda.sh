#!/usr/bin/env bash
cd "${0%/*}"

PKG_NAME=chainconsumer
USER=samreay

echo "Current tag is $TRAVIS_TAG"

export CONDA_BLD_PATH=~/conda-bld
export VERSION=${TRAVIS_TAG#?}

echo "Version building is $VERSION"

mkdir ${CONDA_BLD_PATH}
conda config --set anaconda_upload no

conda build .
conda convert --platform all $CONDA_BLD_PATH/linux-64/$PKG_NAME-*.tar.bz2 -o $CONDA_BLD_PATH
anaconda -t $CONDA_UPLOAD_TOKEN -u $USER upload $CONDA_BLD_PATH/**/$PKG_NAME-*.tar.bz2 --force