#!/usr/bin/env python

from pyveg.coordinates import coordinate_store

#Define directory to save all outputs
output_dir = '/tmp/newtestoutput'

# modify this line to set coords based on entries in `coordinates.py`
coordinate_id = '00'

# parse selection. Note (long, lat) GEE convention.
entry = coordinate_store.loc[coordinate_id]
coordinates = (entry.longitude, entry.latitude)


# date range for Copernicus
#date_range = ['2015-01-01', '2020-01-01']
date_range = ['2016-01-01', '2016-03-01']

#date range for landsat 5
#date_range = ('1988-01-01', '2003-01-01')


# collections for Copernicus
collections_to_use = ['Sentinel2', 'ERA5']

# collections to use for old Landsat
#collections_to_use = ['Landsat4','Landsat5']


do_network_centrality = True

data_collections = {
    'Sentinel2' : {
        'collection_name': 'COPERNICUS/S2',
        'data_type': 'vegetation',
        'RGB_bands': ['B4','B3','B2'],
        'NIR_band': 'B8',
        'mask_cloud': True,
        'cloudy_pix_frac': 50,
        'cloudy_pix_flag': 'CLOUDY_PIXEL_PERCENTAGE',
        'do_network_centrality': do_network_centrality,
        'time_per_point': "1m"
    },
    'Landsat8' : {
        'collection_name': 'LANDSAT/LC08/C01/T1_SR',
        'data_type': 'vegetation',
        'RGB_bands': ['B4','B3','B2'],
        'NIR_band': 'B5',
        'cloudy_pix_flag': 'CLOUD_COVER',
        'do_network_centrality': do_network_centrality,
        'time_per_point': "3m"
    },
    'Landsat5' : {
        'collection_name': 'LANDSAT/LT05/C01/T1_SR',
        'data_type': 'vegetation',
        'RGB_bands': ['B3','B2','B1'],
        'NIR_band': 'B4',
        'cloudy_pix_flag': 'None',
        'do_network_centrality': do_network_centrality,
        'time_per_point': "3m"
    },
    'Landsat4' : {
        'collection_name': 'LANDSAT/LT04/C01/T1_SR',
        'data_type': 'vegetation',
        'RGB_bands': ['B3','B2','B1'],
        'NIR_band': 'B4',
        'cloudy_pix_flag': 'None',
        'do_network_centrality': do_network_centrality,
        'time_per_point': "3m"
    },
    'ERA5' : {
        'collection_name': 'ECMWF/ERA5/MONTHLY',
        'data_type': 'weather',
        'precipitation_band': ['total_precipitation'],
        'temperature_band': ['mean_2m_air_temperature'],
        'time_per_point': "1m"
    }
}

data_collections = {key : value for key,value in data_collections.items() if key in collections_to_use}
