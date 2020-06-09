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
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "Ada"
copyright = "2020, Anne-Marie Tousch, Christophe Renaudin"
author = "Anne-Marie Tousch, Christophe Renaudin"

master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Exclude docstrings from external base classes
autodoc_mock_imports = ["pytorch_lightning", "streamlit"]


# Workaround to not skip ADA classes that inherit from (mocked) classes from skipped modules
def setup(app):
    def skip_member(app, what, name, obj, skip, options):
        if getattr(obj, "__sphinx_mock__", False) and obj.__module__.startswith("ada."):
            if (
                obj.__new__.__module__ is not None
                and obj.__new__.__module__.startswith("sphinx.")
            ):
                # Trick to keep constructor arguments
                obj.__new__ = object.__new__
            return False
        return None

    app.connect("autodoc-skip-member", skip_member)


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"
