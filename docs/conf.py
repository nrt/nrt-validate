# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nrt-validate'
copyright = 'European Union, 2024, Loïc Dutrieux, Keith Araño & Jonas Viehweger'
author = 'Loic Dutrieux, Keith Arano, Jonas Viehweger'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.mathjax'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'gallery/README.rst']

# Gallery configuration
sphinx_gallery_conf = {
     'filename_pattern': '/plot_',
     'examples_dirs': 'gallery',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
    'style_nav_header_background': "#f8efc8"
}
