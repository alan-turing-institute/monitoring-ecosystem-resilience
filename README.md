[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/urbangrammarai/gee_pipeline/master?labpath=notebooks)


# Google Earth Engine Pipeline

This repository is a pipeline for retrieving imagery from Google Earth Engine for the [Urban Grammar](https://urbangrammarai.xyz/) project.

The purpose is to obtain a time-series (ideally at least yearly) of cloud-free composite images for the UK, for use to infer spatial signatures.

The repo is a fork of [monitoring-ecosystem-resilience](https://github.com/alan-turing-institute/monitoring-ecosystem-resilience) because of the functional similarity, despite the differences in purpose.

The code in this repository is intended to perform three inter-related tasks:

* Download and process satellite imagery from Google Earth Engine.
* Generate cloud-free composite images from Sentinel-2 for each year since 2016.
* Generate "chips" (or patches) suitable for input into the inference model.



### Python

The tasks above are all implemented in Python in the *peep* package. See the [README.md](peep/README.md) in the `peep` subdirectory for details on installation and usage.
