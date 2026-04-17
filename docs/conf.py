"""Sphinx configuration for KempnerForge documentation."""

from __future__ import annotations

import importlib.metadata

project = "KempnerForge"
author = "Kempner Institute"
copyright = "2026, Kempner Institute"

try:
    release = importlib.metadata.version("kempnerforge")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_class_signature = "separated"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "furo"
html_title = f"KempnerForge {version}"
html_static_path = ["_static"]
html_logo = "assets/logo.png"
html_favicon = None
html_theme_options = {
    "source_repository": "https://github.com/KempnerInstitute/KempnerForge",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/KempnerInstitute/KempnerForge",
            "class": "",
        },
    ],
}

nitpicky = False

# PyTorch's DeviceMesh has an unresolved ``ArrayLike`` forward reference in
# its signature — that is upstream, not ours, so suppress those warnings so
# ``-W`` in CI still catches real issues in our own code.
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]
