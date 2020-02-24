#!/usr/bin/env python

output_dir = 'test'

coordinates = (27.99,11.29) # initial
#coordinates = (28.37,11.12) # labyrinths
 
date_range = ('2016-01-01', '2017-01-01')

num_days_per_point = 30

data_sources = {
    'COPERNICUS/S2': {
        'data_type': 'vegetation',
        'RGB_bands': ('B4','B3','B2'),
        'NIR_band': 'B8',
        'cloudy_pix_flag': 'CLOUDY_PIXEL_PERCENTAGE'
    },
    'LANDSAT/LC08/C01/T1_SR': {
        'data_type': 'vegetation',
        'RGB_bands': ('B4','B3','B2'),
        'NIR_band': 'B5',
        'cloudy_pix_flag': 'CLOUD_COVER'
    },
    'NOAA/PERSIANN-CDR': {
        'data_type': 'weather',
        'precipitation_band': 'precipitation'
    },
    'NASA/GPM_L3/IMERG_V06': {
        'data_type': 'weather',
        'precipitation_band': 'precipitationCal'
    },
    'ECMWF/ERA5/MONTHLY': {
        'data_type': 'weather',
        'precipitation_band': 'total_precipitation',
        'temperature_band': 'mean_2m_air_temperature'
    }
}

cloudy_pixel_percent = 10