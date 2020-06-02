"""
Tests for the modules that process the images downloaded from GEE
"""

import os
import pytest
import json
import shutil

from pyveg.src.processor_modules import (
    VegetationImageProcessor,
    NetworkCentralityCalculator,
    WeatherImageToJSON
)


def test_Sentinel2_image_processor():
    """
    Should combine tif files into RGB, NDVI, and BWNDVI
    big images, and split RGB and BWNVI into sub-images.
    """
    dir_path = os.path.join(os.path.dirname(__file__), "..", "testdata", "Sentinel2", "test_tif")
    tmp_png_path = os.path.join(os.path.dirname(__file__), "..", "testdata", "Sentinel2", "tmp_png")

    vip = VegetationImageProcessor()
    vip.input_location = dir_path
    vip.output_location = tmp_png_path
    vip.coords = [11.58,27.95]
    vip.configure()
    vip.run()
    assert os.path.exists(os.path.join(tmp_png_path, "2018-03-01", "PROCESSED"))
    assert len(os.listdir(os.path.join(tmp_png_path, "2018-03-01", "PROCESSED"))) == 3
    assert os.path.exists(os.path.join(tmp_png_path, "2018-03-01", "SPLIT"))
    assert len(os.listdir(os.path.join(tmp_png_path, "2018-03-01", "SPLIT"))) == 1452
    shutil.rmtree(tmp_png_path, ignore_errors=True)



def test_ERA5_image_to_json():
    """
    Get values out of tif files and put into JSON file.
    """
    dir_path = os.path.join(os.path.dirname(__file__), "..", "testdata", "ERA5", "test_tif")
    tmp_json_path = os.path.join(os.path.dirname(__file__), "..", "testdata", "ERA5", "tmp_json")

    wip = WeatherImageToJSON()
    wip.input_location = dir_path
    wip.output_location = tmp_json_path
    wip.coords = [11.58,27.95]
    wip.configure()
    wip.run()
    assert os.path.exists(os.path.join(tmp_json_path, "2016-01-16","JSON","WEATHER", "weather_data.json"))
    results = json.load(open(os.path.join(tmp_json_path, "2016-01-16","JSON", "WEATHER", "weather_data.json")))
#    assert "2016-01-16" in results.keys()
    assert "mean_2m_air_temperature" in results.keys()
    assert "total_precipitation" in results.keys()
    assert isinstance(results["mean_2m_air_temperature"], float)
    assert isinstance(results["total_precipitation"], float)
    shutil.rmtree(tmp_json_path)



def test_network_centrality_calculator():
    """
    Test that we can go from a directory containing some 50x50 BWNVI images
    to a json file containing network centrality values.
    """
    dir_path = os.path.join(os.path.dirname(__file__), "..", "testdata", "Sentinel2", "test_png")
    tmp_json_path = os.path.join(os.path.dirname(__file__), "..", "testdata", "Sentinel2", "tmp_json")
    ncc = NetworkCentralityCalculator()
    ncc.input_location = dir_path
    ncc.output_location = tmp_json_path
    ncc.configure()
    ncc.run()
    assert os.path.exists(os.path.join(tmp_json_path, "2018-03-01","JSON","NC","network_centralities.json"))
    nc_json = json.load(open(os.path.join(tmp_json_path, "2018-03-01","JSON","NC","network_centralities.json")))
    assert isinstance(nc_json, list)
    assert isinstance(nc_json[0], dict)
    # test float values
    for key in ["latitude", "longitude", "offset50", "ndvi", "ndvi_veg"]:
        assert key in nc_json[0].keys()
        assert isinstance(nc_json[0][key], float)
        assert nc_json[0][key] != 0.
    assert "date" in nc_json[0].keys()
    assert isinstance(nc_json[0]["date"], str)
    shutil.rmtree(tmp_json_path)
