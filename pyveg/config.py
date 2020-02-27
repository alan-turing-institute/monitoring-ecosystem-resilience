#!/usr/bin/env python

output_dir = 'TEST_OUT'

coordinates = (27.99,11.29) # initial
#coordinates = (28.37,11.12) # labyrinths
 
date_range = ('2016-01-01', '2017-01-01')

num_days_per_point = 30

collections_to_use = ['unsupported']

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
    'Landsat' : {
        'collection_name': 'LANDSAT/LC08/C01/T1_SR',
        'type': 'vegetation',
        'RGB_bands': ('B4','B3','B2'),
        'NIR_band': 'B5',
        'cloudy_pix_flag': 'CLOUD_COVER',
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
        'temperature_band': ['probabilityLiquidPrecipitation']
    },
    'unsupported' : {
        'collection_name': "ECMWF/ERA5/DAILY",
        'type': 'weather',
        'precipitation_band': ['total_precipitation'],
        'temperature_band': ['mean_2m_air_temperature']
    }
}

data_collections = {key : value for key,value in data_collections.items() if key in collections_to_use}

cloudy_pixel_percent = 10

