#!/usr/bin/env python
import os

#Define directory to save all outputs
output_dir_ = '/Users/crangelsmith/PycharmProjects/GEE_DATA/'
name = "Gaps_Sudan_ERA"

coordinates = (28.198,10.96)

# date range for Copernicus
date_range = ('2015-01-01', '2020-01-01')


# collections for Copernicus
collections_to_use = ['ERA5']

# collections to use for old Landsat
#collections_to_use = ['Landsat4','Landsat5']

output_dir_name = name+"_"+str(coordinates[0])+"_"+str(coordinates[1])+"_"+str(date_range[0])+"_"+str(date_range[1])

output_dir = os.path.join(output_dir_,output_dir_name)


do_network_centrality = True

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

data_collections = {key : value for key,value in data_collections.items() if key in collections_to_use}

# not currently used
# cloudy_pixel_percent = 10
