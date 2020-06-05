#!/usr/bin/env python

"""
Configuration file to set parameters for GEE download jobs. This file is 
used to define job options when running the `pyveg_gee_download` command.

"""

from pyveg.coordinates import coordinate_store

# user specified output dir
output_dir = 'X'

# modify this line to set coords based on entries in `coordinates.py`
coordinate_id = '00'

# parse selection. Note (long, lat) GEE convention.
entry = coordinate_store.loc[coordinate_id]
coordinates = (entry.longitude, entry.latitude)

# if you want to test new coordinates without adding them to the store
# you may overwrite the coordinates here. Note the (long, lat) GEE convention.
#coordinates = (long, lat)

# date range for Copernicus
date_range = ('2015-01-01', '2020-04-01')

# collections for Sentinel2
collections_to_use = ['Sentinel2', 'ERA5']

# date range for landsat 4/5. In this range there is relatively good data
#date_range = ('1988-01-01', '2003-01-01')
# collections to use for old Landsat
#collections_to_use = ['Landsat4','Landsat5']

# turn off to quickly scout out new locations
do_network_centrality = True

# parameter dictionary for different GEE collections
data_collections = {
    'Sentinel2' : {
        'collection_name': 'COPERNICUS/S2',
        'type': 'vegetation',
        'RGB_bands': ('B4','B3','B2'),
        'NIR_band': 'B8',
        'cloudy_pix_flag': 'CLOUDY_PIXEL_PERCENTAGE',
        'do_network_centrality': do_network_centrality,
        'num_days_per_point': 30
    },
    'Landsat8' : {
        'collection_name': 'LANDSAT/LC08/C01/T1_SR',
        'type': 'vegetation',
        'RGB_bands': ('B4','B3','B2'),
        'NIR_band': 'B5',
        'cloudy_pix_flag': 'CLOUD_COVER',
        'do_network_centrality': do_network_centrality,
        'num_days_per_point': 90
    },
    'Landsat5' : {
        'collection_name': 'LANDSAT/LT05/C01/T1_SR',
        'type': 'vegetation',
        'RGB_bands': ('B3','B2','B1'),
        'NIR_band': 'B4',
        'cloudy_pix_flag': 'None',
        'do_network_centrality': do_network_centrality,
        'num_days_per_point': 90
    },
    'Landsat4' : {
        'collection_name': 'LANDSAT/LT04/C01/T1_SR',
        'type': 'vegetation',
        'RGB_bands': ('B3','B2','B1'),
        'NIR_band': 'B4',
        'cloudy_pix_flag': 'None',
        'do_network_centrality': do_network_centrality,
        'num_days_per_point': 90
    },
    'ERA5' : {
        'collection_name': 'ECMWF/ERA5/MONTHLY',
        'type': 'weather',
        'precipitation_band': ['total_precipitation'],
        'temperature_band': ['mean_2m_air_temperature'],
        'num_days_per_point': 30
    }
}

# select only those collections we want to use in this job
data_collections = {key : value for key,value in data_collections.items() if key in collections_to_use}
