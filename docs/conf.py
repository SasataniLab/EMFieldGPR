# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EMFieldML"
#copyright = "2024, Takuya Sasatani"
#author = "Takuya Sasatani"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "myst_parser",  # Add MyST parser for Markdown support
    "sphinx_click",  # Add sphinx-click for Click CLI documentation
]

# Add .md files to source_suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Theme options
html_theme_options = {
    "repository_url": "https://github.com/SasataniLab/EMFieldML",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "primary_sidebar_end": ["theme-switcher"],
    "navbar_end": ["theme-switcher"],
    "show_toc_level": 2,
    "use_source_button": False,  # Disable source code links
}

# -- Options for autodoc extension ---------------------------------------
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Suppress warnings for missing imports
autodoc_mock_imports = [
    "numpy", "scipy", "matplotlib", "sklearn", "torch", "gpytorch",
    "polyscope", "open3d", "pyvista", "numpy-stl", "scikit-rf", "skrf",
    "coloredlogs", "python-dotenv", "click", "gdown", "cmake", "seaborn",
    "stl", "mesh", "tqdm", "dotenv", "bisect", "csv", "math", "pathlib",
    "statistics", "datetime", "tempfile", "subprocess", "threading", "time",
    "copy", "os", "sys", "collections", "collections.abc",
]

# Don't execute code during import - mock internal config objects that cause issues
autodoc_mock_imports.extend([
    "EMFieldML.config.config", "EMFieldML.config.paths", "EMFieldML.config.template",
    "EMFieldML.config", "config", "paths", "template"
])

# -- Options for intersphinx extension -----------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- Options for todo extension ------------------------------------------
todo_include_todos = True


# -- Path setup ----------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
