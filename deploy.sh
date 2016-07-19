#!/usr/bin/env bash
if [ "$TRAVIS_PULL_REQUEST" != "false" -o "$TRAVIS_BRANCH" != "master" ]; then
    echo "Not on master branch, or pull request. Not building doco"
    exit 0;
fi
if [ -n "$GITHUB_API_KEY2" ]; then
    echo "Github key found. Building documentation."
    cd "$TRAVIS_BUILD_DIR"/doc
    make clean
    make html
    make html
    cd "$TRAVIS_BUILD_DIR"
    rm -rf .git/
    cd doc/out/html
    git config --global user.email "travis"
    git config --global user.name "travis"
    touch .nojekyll
    git init
    git add .
    echo "Committing"
    git commit -m init
    # Make sure to make the output quiet, or else the API token will leak!
    # This works because the API key can replace your password.
    echo "Pushing"
    git push -f -q "https://${GITHUB_API_KEY2}@${GH_REF}" master:gh-pages  > /dev/null 2>&1 && echo "Pushed"
fi
echo "Deploy script ending"