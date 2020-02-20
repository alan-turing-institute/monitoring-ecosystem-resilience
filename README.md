# monitoring-ecosystem-resilience
Repository for mini-projects in the Data science for Sustainable development project.

Currently the focus of code in this repository is understanding vegetation patterns in semi-arid environments.

The code in this repository is intended to perform three inter-related tasks:
* Download and process satellite imagery from Google Earth Engine.
* Generate simulated vegetation patterns.
* Calculate graph metrics to quantify the interconnectedness of vegetation in real and simulated images.

### Python Setup

The tasks above are all implemented in Python in the *pyveg* package. See the [README.md](pyveg/README.md) in the `pyveg` subdirectory for details on installation and usage.

### R

The pattern-generation and graph-modelling are implemented in R in the `rveg` package.  See the [README.md](rveg/README.md) in that directory for further details.
