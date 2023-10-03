.PHONY: tests

install:
	pip install -U pip poetry -q
	poetry install --with=dev,test --all-extras
	poetry run pre-commit install
	poetry run pre-commit autoupdate

precommit:
	poetry run pre-commit run --all-files

test:
	poetry run pytest

tests: test

all: precommit tests