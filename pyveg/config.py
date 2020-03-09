#!/usr/bin/env python

#Define directory to save all outputs
output_dir = 'TEST'

#coordinates = (27.99,11.29) # dense spots
#coordinates = (28.37,11.12) # labyrinths
#coordinates = (23.54,11.34) # beautiful rivers
coordinates = (27.94,11.58) # spots

date_range = ('2011-01-01', '2012-01-01')

collections_to_use = ['Copernicus', 'ERA5']

do_network_centrality = True

data_collections = {
    'Copernicus' : {
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
        'num_days_per_point': 182
    },
    'Landsat5' : {
        'collection_name': 'LANDSAT/LT05/C01/T1_SR',
        'type': 'vegetation',
        'RGB_bands': ('B3','B2','B1'),
        'NIR_band': 'B4',
        'cloudy_pix_flag': 'None',
        'do_network_centrality': do_network_centrality,
        'num_days_per_point': 182
    },
    'Landsat4' : {
        'collection_name': 'LANDSAT/LT04/C01/T1_SR',
        'type': 'vegetation',
        'RGB_bands': ('B3','B2','B1'),
        'NIR_band': 'B4',
        'cloudy_pix_flag': 'None',
        'do_network_centrality': do_network_centrality,
        'num_days_per_point': 182
    },
    'ERA5' : {
        'collection_name': 'ECMWF/ERA5/MONTHLY',
        'type': 'weather',
        'precipitation_band': ['total_precipitation'],
        'temperature_band': ['mean_2m_air_temperature'],
        'num_days_per_point': 30
    }
}

data_collections = {key : value for key,value in data_collections.items() if key in collections_to_use}

# not currently used
# cloudy_pixel_percent = 10 
