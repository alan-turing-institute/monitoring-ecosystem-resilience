#!/usr/bin/env python

output_dir = 'test'

coordinates = (27.99,11.29) # initial
#coordinates = (28.37,11.12) # labyrinths
 
date_range = ('2016-01-01', '2017-01-01')

num_days_per_point = 30

data_sources = [
    {
        'collection_name': 'COPERNICUS/S2',
        'type': 'vegetation',
        'RGB_bands': ('B4','B3','B2'),
        'NIR_band': 'B8',
        'cloudy_pix_flag': 'CLOUDY_PIXEL_PERCENTAGE'
    },
    {
        'collection_name': 'LANDSAT/LC08/C01/T1_SR',
        'type': 'vegetation',
        'RGB_bands': ('B4','B3','B2'),
        'NIR_band': 'B5',
        'cloudy_pix_flag': 'CLOUD_COVER'
    },
    {
        'collection_name': 'NOAA/PERSIANN-CDR',
        'type': 'precipitation',
        'precipitation_band': 'precipitation'
    },
    {
        'collection_name': 'NASA/GPM_L3/IMERG_V06',
        'type': 'precipitation',
        'precipitation_band': 'precipitationCal'
    },
    {
        'collection_name': 'ECMWF/ERA5/MONTHLY',
        'type': 'unsupported',
        'precipitation_band': 'total_precipitation',
        'temperature_band': 'mean_2m_air_temperature'
    }
]

cloudy_pixel_percent = 10