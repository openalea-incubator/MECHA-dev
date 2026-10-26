"""Sphinx configuration of MECHA's documentation: `python -m sphinx -b html doc doc/_build/html`."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# The tutorials (example/tutorials/user and developer), rendered from their stored outputs: copied at each build
import shutil
for series in ("user", "developer"):
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks", series)
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "example", "tutorials", series),
                    target, ignore=shutil.ignore_patterns(".ipynb_checkpoints", "*.py", "*.csv", "outputs"))
nbsphinx_execute = "never"

project = "MECHA"
author = "the MECHA authors"
extensions = [
    "sphinx.ext.autodoc",          # docstrings
    "sphinx.ext.mathjax",          # equations
    "sphinx.ext.viewcode",         # links to the source
    "myst_parser",                 # Markdown pages (the README)
    "openalea.metafspm.sphinx",    # metafspm components rendered from their declarations
    "nbsphinx",                    # the tutorials' notebooks
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
