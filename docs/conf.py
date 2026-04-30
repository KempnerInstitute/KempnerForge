"""Sphinx configuration for KempnerForge documentation."""

from __future__ import annotations

import importlib.metadata

project = "KempnerForge"
author = "Kempner Institute for the Study of Natural and Artificial Intelligence at Harvard University"

try:
    release = importlib.metadata.version("kempnerforge")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

copyright = f"2026, Kempner Institute for the Study of Natural and Artificial Intelligence at Harvard University · v{release}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_mock_imports = [
    "torch",
    "torchao",
    "torchdata",
    "datasets",
    "transformers",
    "wandb",
    "tensorboard",
    "sympy",
    "numpy",
    "triton",
]
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
    "torch": ("https://pytorch.org/docs/main", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "furo"
html_title = "KempnerForge"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "assets/logo.png"
html_favicon = None
html_theme_options = {
    "sidebar_hide_name": True,
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

# Autosummary generates stubs for both package re-exports (e.g. kempnerforge.config.JobConfig)
# and their defining submodules (kempnerforge.config.job.JobConfig). Docstring refs to the
# short name then resolve to two targets; silence those ambiguities.
suppress_warnings = ["ref.python"]
