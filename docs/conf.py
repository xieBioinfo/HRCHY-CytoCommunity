# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os, sys

# DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
# ROOT_DIR = os.path.abspath(os.path.join(DOCS_DIR, ".."))
# SRC_DIR  = os.path.join(ROOT_DIR, "src")

# # make Sphinx able to import hrchy_cytocommunity
# if SRC_DIR not in sys.path:
#     sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./notebooks'))
sys.path.insert(0, os.path.abspath('../src'))
print("[conf] sys.path[0] =", sys.path[0])
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HRCHY-CytoCommunity'
copyright = '2025, RunzhiXie'
author = 'RunzhiXie'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_nb',
              "sphinx.ext.autodoc",
                "sphinx.ext.autosummary",
                "sphinx.ext.napoleon",
              ]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',  # 统一使用 myst-nb
}
autosummary_generate = True
# myst-nb 配置
nb_execution_mode = "off"  # 不执行笔记本


templates_path = ['_templates']
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
