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



## Using the Docker image

There is a Docker image with `peep` and its dependencies preinstalled.




### Authenticating the Google Earth Engine


It is not possible authenticate the google earth engine client directly from with the docker container

In a shell in the docker container:

`earthengine authenticate --quiet`


On your own computer

```
gcloud auth application-default login --remote-bootstrap="https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.apps.googleusercontent.com&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fearthengine+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdevstorage.full_control+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Faccounts.reauth&state=aSWZIEfr47wX483XfpU8EbT2kp1oQG&access_type=offline&code_challenge=yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy&code_challenge_method=S256&token_usage=remote"
DO NOT PROCEED UNLESS YOU ARE BOOTSTRAPPING GCLOUD ON A TRUSTED MACHINE WITHOUT A WEB BROWSER AND THE ABOVE COMMAND WAS THE OUTPUT OF `gcloud auth application-default
login --no-browser` FROM THE TRUSTED MACHINE.
```

<-- Add screenshots here -->


details are saved here:
[/root/.config/gcloud/application_default_credentials.json]


You should be able to run this command without any errors (no output indicates success)
```
python -c "import ee; ee.Initialize()"
```
