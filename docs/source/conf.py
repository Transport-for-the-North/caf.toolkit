# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


from __future__ import annotations

# Built-Ins
import importlib
import inspect
import os
import re
import shutil
import sys
from pathlib import Path

# -- Clean-up generated ------------------------------------------------------
generated_folder = Path(__file__).parent / "_generated"
if generated_folder.is_dir():
    shutil.rmtree(generated_folder)

# -- Path setup --------------------------------------------------------------
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
    "sphinx.ext.linkcode",
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
autodoc_typehints = "description"

# Auto summary options
autosummary_generate = True
autosummary_imported_members = True
modindex_common_prefix = ["caf.", "caf.toolkit."]

# -- Options for Sphinx Examples gallery -------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "_generated/examples",  # path to where to save gallery generated output
    "backreferences_dir": "_generated/examples/backrefs",  # path to the backreferences files
    "doc_module": ("caf.toolkit",),
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
html_show_sourcelink = False

master_doc = "index"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "use_edit_page_button": True,
    "logo": {
        "text": f"{project} {version}",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Transport-for-the-North/caf.toolkit",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "header_links_before_dropdown": 4,
    "external_links": [
        {
            "name": "Changelog",
            "url": "https://github.com/Transport-for-the-North/caf.toolkit/releases",
        },
        {
            "name": "Issues",
            "url": "https://github.com/Transport-for-the-North/caf.toolkit/issues",
        },
        {
            "name": "CAF Handbook",
            "url": "https://transport-for-the-north.github.io/CAF-Handbook/",
        },
    ],
    "primary_sidebar_end": ["indices.html", "sidebar-ethical-ads.html"],
}
html_context = {
    "github_url": "https://github.com",
    "github_user": "Transport-for-the-North",
    "github_repo": "caf.toolkit",
    "github_version": "main",
    "doc_path": "docs/source",
}

# -- Options for Linkcode extension ------------------------------------------


def _get_object_filepath(module: str, fullname: str) -> str:
    """Get filepath (including line numbers) for object in module."""
    mod = importlib.import_module(module)
    if "." in fullname:
        objname, attrname = fullname.split(".")
        obj = getattr(mod, objname)

        try:
            # object is method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is attribute of a class so use class
            obj = getattr(mod, objname)

    else:
        try:
            obj = getattr(mod, fullname)
        except AttributeError:
            return module.replace(".", "/") + ".py"

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
        filepath = f"{file}#L{lines[1]}"
    except (TypeError, OSError):
        filepath = module.replace(".", "/") + ".py"

    return filepath


def linkcode_resolve(domain: str, info: dict) -> str | None:
    """Resolve URLs for linking to code on GitHub.

    See sphinx.ext.linkcode extension docs for more details
    https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html
    """
    if domain != "py":
        return None
    if not info["module"]:
        return None

    filepath = _get_object_filepath(info["module"], info["fullname"])
    # Check if path is in the directory
    try:
        filepath = str(Path(filepath).relative_to(dir_path))
    except ValueError:
        return None

    tag = f"v{version.split('+', maxsplit=1)[0]}"
    github_url = (
        f"{html_context['github_url']}/{html_context['github_user']}"
        f"/{html_context['github_repo']}/tree/{tag}"
    )

    return f"{github_url}/{filepath}"
