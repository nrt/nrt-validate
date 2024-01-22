#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
from setuptools import setup, find_packages
import os

# Parse the version from the main __init__.py
with open('nrt/validate/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue


with codecs.open('README.rst', encoding='utf-8') as f:
    readme = f.read()

extra_reqs = {'dask': ['dask'],
              'docs': ['sphinx',
                       'sphinx_rtd_theme',
                       'matplotlib',
                       'sphinx-gallery']}

setup(name='nrt-validate',
      version=version,
      description=u"Validation of alerting system results produced by tools like nrt",
      long_description_content_type="text/x-rst",
      long_description=readme,
      keywords='sentinel2, xarray, validation, forest, monitoring, change',
      author=u"Loic Dutrieux, Keith Arano, Jonas Viehweger",
      author_email='loic.dutrieux@ec.europa.eu',
      url='https://code.europa.eu/jrc-forest/nrt',
      license='EUPL-v1.2',
      classifiers=[
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
      ],
      packages=find_packages(),
      install_requires=[
          'xarray',
          'ipywidget',
          'ipyleaflet',
          'bqplot',
          'ipyevents',
          'PIL',
          'rioxarray',
          'netCDF4',
          'pandas'
      ],
      python_requires=">=3.9",
      extras_require=extra_reqs)

