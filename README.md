![Build status](https://api.travis-ci.com/alan-turing-institute/monitoring-ecosystem-resilience.svg?branch=master)

# monitoring-ecosystem-resilience
Repository for mini-projects in the Data science for Sustainable development project.

Currently the focus of code in this repository is understanding vegetation patterns in semi-arid environments.

The code in this repository is intended to perform three inter-related tasks:
* Download and process satellite imagery from Google Earth Engine.
* Generate simulated vegetation patterns.
* Calculate graph metrics to quantify the interconnectedness of vegetation in real and simulated images.

### Python

The tasks above are all implemented in Python in the *pyveg* package.  This can be installed by doing
```
pip install .
```
from this directory.  See the README.md in the `pyveg` subdirectory for further details.

### R

The pattern-generation and graph-modelling are implemented in R in the `rveg` package.  See the README.md in that directory for further details.
