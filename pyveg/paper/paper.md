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
linked to the resiliance of the ecosystem [@Mander:2017, @Trichon:2018]. 
Using remote sensing techniques,  vegetation patterns in these regions 
can be studied, and an analysis of the resiliance of the ecosystem can 
be performed.

This package implements functionality to download and process data
from Google Earth Engine (GEE), in the context of performing such a 
resiliance analysis.

Google Earth Engine Editor scripts are also provided to help 
researchers discover locations of ecosystems which may be in
decline.


# Downloading data from Google Earth Engine

In order to interact with the GEE API, the user must sign up to GEE 
and obtain an API key. Upon running `pyveg` for the first time, the 
user will be prompted to enter their API key. The `run_pyveg_pipeline`
command allows the user to initiate a download job, which is configured
using a configuration file. The command accepts an argument, `--config_file`, 
which is the filename of the configuration file to use for the download.

Within the configuration file, the user can specify the following:
- Coordinates of the location.
- Start and end dates of the time series.
- Frequency with which to sample.
- GEE collections to download from (currently vegetation and precipitation
  collections are supported).

`pyveg` will then form a series of date ranges, and query GEE for the relevant
data in each date range. Colour (RGB) and Normalised Difference vegetation
Index (NDVI) images are downloaded from vegetation collections. For precipitation
and temperature information, `pyveg` defaults to using the ERA5 GEE collection.


# Network centrality metrics

After completetion of the download job, `pyveg` computes the network centrality 
of the vegetation [@Mander:2017]. To do this, the downloaded NDVI image is thresholded.


# Time series analysis 


# Acknowledgements

We acknowledge contributions and support from Tim Lenton, Chris Boulton, 
and Jessie Abrams during the course of this project.

# References