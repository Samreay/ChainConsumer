.PHONY: tests docs build

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
	poetry run mkdocs build

pushdocs:
	poetry run mkdocs gh-deploy --force

build:
	poetry publish --build --dry-run

publish:
	poetry publish --build

tests: test

all: precommit tests