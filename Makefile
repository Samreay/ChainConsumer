.PHONY: tests docs build
VERSION := $(shell git for-each-ref refs/tags --format='%(refname:short)' | grep -E "^v?[0-9]+\..*" | tail -n1)

install:
	pip install -U pip poetry -q
	poetry install --with=dev,test --all-extras
	poetry run pre-commit install
	poetry run pre-commit autoupdate

precommit:
	poetry run pre-commit run --all-files

test:
	poetry run pytest

serve:
	rm -rf docs/generated/gallery;
	poetry run mkdocs serve --clean

docs:
	poetry run poetry version $(VERSION) && poetry run mkdocs build

pushdocs:
	poetry run poetry version $(VERSION) && poetry run mkdocs gh-deploy --force

build:
	rm -rf dist; poetry version $(VERSION) && poetry publish --build --dry-run

publish:
	rm -rf dist; poetry config pypi-token.pypi $$PYPI_TOKEN && poetry version $(VERSION) && poetry publish --build

tests: test

all: precommit tests