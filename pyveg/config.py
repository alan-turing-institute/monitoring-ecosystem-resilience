#!/usr/bin/env python

#Define directory to save all outputs
output_dir = 'RUN1'

#coordinates = (27.99,11.29) # initial
coordinates = (28.37,11.12) # labyrinths
 
date_range = ('2015-01-01', '2020-001-01')

num_days_per_point = 90

collections_to_use = ['Copernicus','Landsat8', 'ERA5']

do_network_centrality = True

data_collections = {
    'Copernicus' : {
        'collection_name': 'COPERNICUS/S2',
        'type': 'vegetation',
        'RGB_bands': ('B4','B3','B2'),
        'NIR_band': 'B8',
        'cloudy_pix_flag': 'CLOUDY_PIXEL_PERCENTAGE',
        'do_network_centrality': do_network_centrality
    },
    'Landsat8' : {
        'collection_name': 'LANDSAT/LC08/C01/T1_SR',
        'type': 'vegetation',
        'RGB_bands': ('B4','B3','B2'),
        'NIR_band': 'B5',
        'cloudy_pix_flag': 'CLOUD_COVER',
        'do_network_centrality': do_network_centrality
    },
    'Landsat5' : {
        'collection_name': 'LANDSAT/LT05/C01/T1_SR',
        'type': 'vegetation',
        'RGB_bands': ('B3','B2','B1'),
        'NIR_band': 'B4',
        'cloudy_pix_flag': 'None',
        'do_network_centrality': do_network_centrality
    },
    'Landsat4' : {
        'collection_name': 'LANDSAT/LT04/C01/T1_SR',
        'type': 'vegetation',
        'RGB_bands': ('B3','B2','B1'),
        'NIR_band': 'B4',
        'cloudy_pix_flag': 'None',
        'do_network_centrality': do_network_centrality
    },
    'NOAA' : {
        'collection_name': 'NOAA/PERSIANN-CDR',
        'type': 'weather',
        'precipitation_band': ['precipitation']
    },
    'NASA' : {
        'collection_name': 'NASA/GPM_L3/IMERG_V06',
        'type': 'weather',
        'precipitation_band': ['precipitationCal'],
    },
    'ERA5' : {
        'collection_name': 'ECMWF/ERA5/DAILY',
        'type': 'weather',
        'precipitation_band': ['total_precipitation'],
        'temperature_band': ['mean_2m_air_temperature']
    }
}

data_collections = {key : value for key,value in data_collections.items() if key in collections_to_use}

# not currently used
# cloudy_pixel_percent = 10 

