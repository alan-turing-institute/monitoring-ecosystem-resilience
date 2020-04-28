import os
import sys
import shutil


import unittest

from pyveg.src.pyveg_pipeline import Sequence
from pyveg.src.download_modules import VegetationDownloader, WeatherDownloader
from pyveg.src.processor_modules import (
    VegetationImageProcessor,
    NetworkCentralityCalculator,
    WeatherImageToJSON
)

coordinates = (27.99,11.29)

# get one data point for the test
date_range = ['2016-01-01', '2016-01-31']
time_per_point = "30d"

do_network_centrality = True

data_collections = {
    'Copernicus' : {
        'collection_name': 'COPERNICUS/S2',
        'type': 'vegetation',
        'RGB_bands': ['B4','B3','B2'],
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
    'ERA5' : {
        'collection_name': "ECMWF/ERA5/DAILY",
        'type': 'weather',
        'precipitation_band': ['total_precipitation'],
        'temperature_band': ['mean_2m_air_temperature']
    }
}

test_out_dir = 'test_out'
@unittest.skipIf(os.environ.get('TRAVIS') == 'true','Skipping this test on Travis CI.')
def test_get_vegetation():
    s = Sequence("vegetation")
    s.set_config(data_collections["Copernicus"])
    s.output_dir = test_out_dir
    s.coords = coordinates
    s.date_range = date_range
    s.n_sub_images = 1
    s.time_per_point = time_per_point
    s += VegetationDownloader()
    s += VegetationImageProcessor()
    s += NetworkCentralityCalculator()
    s.configure()
    s.run()
    assert(os.path.exists(os.path.join(test_out_dir, "2016-01-16","network_centralities.json")))
    shutil.rmtree(test_out_dir, ignore_errors=True)




@unittest.skipIf(os.environ.get('TRAVIS') == 'true','Skipping this test on Travis CI.')
def test_get_rainfall():
    s = Sequence("weather")
    s.set_config(data_collections["ERA5"])
    s.output_dir = test_out_dir
    s.coords = coordinates
    s.date_range = date_range
    s.time_per_point = time_per_point
    s += WeatherDownloader()
    s += WeatherImageToJSON()
    s.configure()
    s.run()
    assert(os.path.exists(os.path.join(test_out_dir, "RESULTS", "weather_data.json")))
    shutil.rmtree(test_out_dir, ignore_errors=True)

#    from pyveg.src.process_satellite_data import get_vegetation, get_weather
#    print('Warning: this test is expected to take a while to run...')
#    result = get_weather(test_out_dir, data_collections['NOAA'], coordinates, date_range)
#    assert (len(result.items())!=0)
#    shutil.rmtree(test_out_dir, ignore_errors=True)
