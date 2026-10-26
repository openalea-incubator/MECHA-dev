"""Sphinx configuration of MECHA's documentation: `python -m sphinx -b html doc doc/_build/html`."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "MECHA"
author = "the MECHA authors"
extensions = [
    "sphinx.ext.autodoc",          # docstrings
    "sphinx.ext.mathjax",          # equations
    "sphinx.ext.viewcode",         # links to the source
    "myst_parser",                 # Markdown pages (the README)
    "openalea.metafspm.sphinx",    # metafspm components rendered from their declarations
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
