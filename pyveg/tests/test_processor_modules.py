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
    vip.input_dir = dir_path
    vip.output_dir = tmp_png_path
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
    wip.input_dir = dir_path
    wip.output_dir = tmp_json_path
    wip.coords = [11.58,27.95]
    wip.configure()
    wip.run()
    assert os.path.exists(os.path.join(tmp_json_path, "RESULTS", "weather_data.json"))
    results = json.load(open(os.path.join(tmp_json_path, "RESULTS", "weather_data.json")))
    assert "2016-01-16" in results.keys()
    assert "mean_2m_air_temperature" in results["2016-01-16"].keys()
    assert "total_precipitation" in results["2016-01-16"].keys()
    assert isinstance(results["2016-01-16"]["mean_2m_air_temperature"], float)
    assert isinstance(results["2016-01-16"]["total_precipitation"], float)
    shutil.rmtree(tmp_json_path)
