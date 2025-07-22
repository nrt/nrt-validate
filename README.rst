************
nrt-validate
************

*``nrt-validate`` is a companion package to `nrt <https://github.com/ec-jrc/nrt>`_, designed for near real-time monitoring of satellite image time series. It extends nrt's capabilities by providing tools for visual interpretation of time-series data, and for the creation and management of reference/validation datasets.*

Features
========
- Interactive time-series annotation interface built with jupyter widgets
- Spatial sampling for unbiased accuracy assessment
- Various estimator or accuracy (Olofsson et al., 2013; Stehman 2014) and timeliness (Bullocks et al., 2022)

Installation
============
To install nrt-validate, run:

.. code-block:: bash

    pip install nrt-validate

Usage
=====
Here is a basic example to get started with the interactive annotation interface in a Jupyter notebook. 

.. code-block:: python

    from nrt.validate.response import Interface
    from nrt.validate import data

    interface = Interface(cube=data.sentinel2_czechia(), features=samples_czechia())
    interface.display_interface()

