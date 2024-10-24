# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "swydd"
copyright = "2024, Daylin Morgan"
author = "Daylin Morgan"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
autodoc_mock_imports = ["argparse"]
html_theme = "shibuya"
html_theme_options = {
    "announcement": "swydd is still undergoing heavy development",
    "github_url": "https://github.com/daylinmorgan/swydd",
    "nav_links": [
        {"title": "API", "url": "api"},
    ],
}
html_static_path = ["_static"]
