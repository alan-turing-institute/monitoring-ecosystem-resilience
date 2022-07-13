[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/urbangrammarai/gee_pipeline/master?labpath=notebooks)

[![Documentation Status](https://readthedocs.org/projects/pyveg/badge/?version=latest)](https://pyveg.readthedocs.io/en/latest/?badge=latest)

# Google Earth Engine Pipeline

This repository is a pipeline for retrieving imagery from Google Earth Engine for the [Urban Grammar](https://urbangrammarai.xyz/) project.

The purpose is to obtain a time-series (ideally at least yearly) of cloud-free composite images for the UK, for use to infer spatial signatures.

The repo is a fork of [monitoring-ecosystem-resilience](https://github.com/alan-turing-institute/monitoring-ecosystem-resilience) because of the functional similarity, despite the differences in purpose.

The code in this repository is intended to perform three inter-related tasks:

* Download and process satellite imagery from Google Earth Engine.
* Generate cloud-free composite images from Sentinel-2 for each year since 2016.
* Generate "chips" (or patches) suitable for input into the inference model.

For legacy reasons (related to the parent monitoring-ecosystem-resilience repo) the python and R packages are currently named `pyveg` and `rveg` respectively. These names are likely to be changed in due course.


### Python

The tasks above are all implemented in Python in the *pyveg* package. See the [README.md](pyveg/README.md) in the `pyveg` subdirectory for details on installation and usage.

### R

The pattern-generation and graph-modelling are implemented in R in the *rveg* package.  See the [README.md](rveg/README.md) in the `rveg` directory for further details.
