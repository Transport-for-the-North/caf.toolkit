# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# Built-Ins
import os
import re
import sys
from pathlib import Path

dir_path = Path(__file__).parents[2]
source = dir_path / "src"
sys.path.insert(0, os.path.abspath(str(source)))
print(f"Added to path: {source}")

# -- Project information -----------------------------------------------------

project = "caf.toolkit"
copyright = "2023, Transport for the North"
author = "Transport for the North"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Local Imports
# The short X.Y version.
import caf.toolkit

version = str(caf.toolkit.__version__)

# The full version, including alpha/beta/rc tags.
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.intersphinx",
    "sphinxarg.ext",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", "_templates/autosummary"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for API summary -------------------------------------------------
numpydoc_show_class_members = False

# Change autodoc settings
autodoc_member_order = "groupwise"
autoclass_content = "class"
autodoc_default_options = {
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": False,
    "private-members": False,
    "exclude-members": "__module__, __weakref__, __dict__",
}
autodoc_typehints = "description"

# Auto summary options
autosummary_generate = True
modindex_common_prefix = ["caf.", "caf.toolkit."]

# -- Options for Sphinx Examples gallery -------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "examples",  # path to where to save gallery generated output
    # Regex pattern of filenames to be ran so the output can be included
    "filename_pattern": rf"{re.escape(os.sep)}run_.*\.py",
}

# -- Options for Linking to external docs (intersphinx) ----------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}
intersphinx_timeout = 30


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

master_doc = "index"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
