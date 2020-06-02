---
title: 'pyveg: A Python package for analysing the time evolution of patterned vegetation using Google Earth Engine'
tags:
  - Python
  - Ecology
  - Remote sensing
  - Time Series Analysis
  - Early warnings 
authors:
  - name: Nick Barlow
    affiliation: 1
  - name: Camila Rangel Smith
    affiliation: 1
  - name: Samuel Van Stroud
    affiliation: 1, 2
affiliations:
 - name: The Alan Turing Institute
   index: 1
 - name: University College London
   index: 2
date: 01 June 2020
bibliography: paper.bib
---

# Introduction

Periodic vegetation patterns (PVP) arise from the interplay between 
forces that drive the growth and mortality of plants. Inter-plant 
competetion for resources, in particular water, can lead to the 
formation of PVP.

Arid and semi-arid ecosystems may be under threat due to changing
precipitation dynamics driven by macroscopic changes in climate. These
regions disiplay some noteable examples of PVP, for example the "tiger
bush" in West Africa.

The mophology of the periodic pattern has been suggested to be 
linked to the resiliance of the ecosystem [@Mander:2017; @Trichon:2018]. 
Using remote sensing techniques,  vegetation patterns in these regions 
can be studied, and an analysis of the resiliance of the ecosystem can 
be performed.

This package implements functionality to download and process data
from Google Earth Engine (GEE), and to subsequently perform a 
resiliance analysis on the downloaded data. The results can be used
to search for typical early warning signals of an ecological collapse 
[@Dakos:2008].

Google Earth Engine Editor scripts are also provided to help 
researchers discover locations of ecosystems which may be in
decline.


# Downloading data from Google Earth Engine

In order to interact with the GEE API, the user must sign up to GEE 
and obtain an API key. Upon running `pyveg` for the first time, the 
user will be prompted to enter their API key. The `run_pyveg_pipeline`
command initiates the downloading of time series data at a single
coordinate location. The job is configured using a configuration file 
specified by the `--config_file` argument.

Within the configuration file, the user can specify the following:
- Coordinates of the location.
- Start and end dates of the time series.
- Frequency with which to sample.
- GEE collections to download from (currently vegetation and precipitation
  collections are supported).

`pyveg` will then form a series of date ranges, and query GEE for the relevant
data in each date range. Colour (RGB) and Normalised Difference vegetation
Index (NDVI) images are downloaded from vegetation collections. Cloud masking 
logic is included to improve data quality. For precipitation and temperature 
information, `pyveg` defaults to using the ERA5 collection.


# Network centrality metrics

After completetion of the download job, `pyveg` computes the network centrality 
of the vegetation [@Mander:2017]. To achieve this, the NDVI image is broken up 
into smaller $50 \times 50$ pixel sub-images. Each sub-image is then thresholded
using the NDVI pixel intensity, and subgraph connectivity is computed for each
binarized sub-image. The resulting metrics are stored, along with mean NDVI pixel 
intensities for each sub-image.


# Time series analysis 


# Acknowledgements

We acknowledge contributions and support from Tim Lenton, Chris Boulton, 
and Jessie Abrams during the course of this project.

# References