import shutil
import sys

import unittest

coordinates = (27.99,11.29)

# get one data point for the test
date_range = ('2016-01-01', '2016-01-31')
num_days_per_point = 30

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
    },
    'unsupported' : {
        'collection_name': "ECMWF/ERA5/DAILY",
        'type': 'weather',
        'precipitation_band': ['total_precipitation'],
        'temperature_band': ['mean_2m_air_temperature']
    }
}

test_out_dir = 'test_out'
@unittest.skipIf(os.environ.get('TRAVIS') == 'true','Skipping this test on Travis CI.')
def test_get_vegetation():
    from pyveg.src.process_satellite_data import get_vegetation, get_weather
    print('Warning: this test is expected to take a while to run...')
    nc_results = get_vegetation(test_out_dir, data_collections['Copernicus'], coordinates, date_range, n_sub_images=1)
    assert( len(nc_results) != 0 )
    shutil.rmtree(test_out_dir, ignore_errors=True)


@unittest.skipIf(os.environ.get('TRAVIS') == 'true','Skipping this test on Travis CI.')
def test_get_rainfall():
    from pyveg.src.process_satellite_data import get_vegetation, get_weather
    print('Warning: this test is expected to take a while to run...')
    result = get_weather(test_out_dir, data_collections['NOAA'], coordinates, date_range)
    assert (len(result.items())!=0)
    shutil.rmtree(test_out_dir, ignore_errors=True)
