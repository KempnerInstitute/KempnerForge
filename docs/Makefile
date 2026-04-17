# Minimal Sphinx makefile for KempnerForge docs.
#
# Usage (from the repo root, inside the uv env):
#   uv run make -C docs html      # one-shot build
#   uv run make -C docs live      # live-reload server on http://127.0.0.1:8000
#   uv run make -C docs clean     # remove build output

SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html live strict clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

strict:
	@$(SPHINXBUILD) -W --keep-going -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

live:
	@$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--ignore "*_build*" --ignore "*generated*"

clean:
	rm -rf "$(BUILDDIR)" api/generated
