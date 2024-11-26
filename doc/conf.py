import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "rationalpy"
author = "Jonathan Palafoutas"
release = "0.1.1"

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "furo"
