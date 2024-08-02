Quickstart
**********

Introduction
============

``nrt-validate`` is a namespace package that extends `nrt <https://nrt.readthedocs.io/en/latest/index.html>`_
with utilities for accuracy and performance assessment of monitoring algorithms and
their outputs. In particular, it contains:

- A customizable user interface for creating reference data through visual interpretation of spatio-temporal data.
  The interface is largely inspired by the timeSync tool (Cohen, Yang & Kennedy, 2010) [1]_,
  combining elements of spatial and temporal context together with very high-resolution imagery
  in a single interface. This allows an interpreter to fully benefit from the various data dimensions for accurate
  identification of land dynamics.
  The interface is suitable for local deployment on any PC via the `jupyter voilà framework <https://voila.readthedocs.io/en/stable/>`_,
  requiring minimal preparation from the user side (data used as visual support for visual interpretation can be fetched directly from
  a cloud archive thanks to the `STACLoader` loader).
- A sampling utility to create a set of sampling locations for a given sampling design and handle the sample set (subsetting, augmenting, etc).
- Multiple estimators of accuracy.
- The timeliness metric proposed by Bullock et al. (2022) [2]_.


Installation
============

Install the package from PyPI using:

.. code-block::

   pip install nrt-validate

When planning to use the user interface, it is necessary to also install ``voila`` (for serving the interface as a local dashboard) and ``fiona`` (often needed to load features contained in a vector file as an in-memory feature collection).

.. code-block::

   pip install voila fiona

Demo
====

For a quick demo of the interface that does not require any data preparation, you can run:

On Unix/Linux:

.. code-block::

   python -c "from nrt.validate import DemoNotebook; print(DemoNotebook.path())" | xargs -I {} voila {} --theme=dark

On Windows (PowerShell):

.. code-block::

   $notebook_path = python -c "from nrt.validate import DemoNotebook; print(DemoNotebook.path())"
   voila $notebook_path --theme=dark

References
==========

.. [1] Cohen, W. B., Yang, Z., & Kennedy, R., 2010.
       Detecting trends in forest disturbance and recovery using yearly Landsat time series:
       2. TimeSync—Tools for calibration and validation. Remote Sensing of Environment, 114(12), 2911-2924.
       https://doi.org/10.1016/j.rse.2010.07.010
	
.. [2] Bullock, E.L., Healey, S.P., Yang, Z., Houborg, R., Gorelick, N., Tang, X. and Andrianirina, C., 2022.
       Timeliness in forest change monitoring: A new assessment framework demonstrated using Sentinel-1 and a continuous change detection algorithm.
       Remote Sensing of Environment, 276, p.113043.
       https://doi.org/10.1016/j.rse.2022.113043

