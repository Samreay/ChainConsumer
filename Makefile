.PHONY: tests docs build
VERSION := $(shell git for-each-ref refs/tags --format='%(refname:short)' | grep -E "^v[0-9]+\..*" | tail -n1)

install_uv:
	@if [ -f "uv" ]; then echo "Downloading uv" && curl -LsSf https://astral.sh/uv/install.sh | sh; else echo "uv already installed"; fi
	uv self update || true

install_python:
	uv python install

install_deps:
	uv sync --all-extras

install_precommit:
	uv run pre-commit install
	uv run pre-commit gc

update_precommit:
	uv run pre-commit autoupdate
	uv run pre-commit gc

precommit:
	uv run pre-commit run --all-files

test:
	uv run pytest tests

tests: test
install: install_uv install_python install_deps install_precommit

serve:
	rm -rf docs/generated/gallery;
	uv run mkdocs serve --clean

docs:
	uv version $(VERSION) && uv run mkdocs build

pushdocs:
	uv version $(VERSION) && uv run mkdocs gh-deploy --force --no-history

build:
	rm -rf dist; uv version $(VERSION) && uv build && uv publish --dry-run

publish:
	rm -rf dist; uv version $(VERSION) && uv build && uv publish

tests: test

all: precommit tests