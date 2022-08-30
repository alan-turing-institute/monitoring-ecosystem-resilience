"""
Tests for the modules that process the images downloaded from GEE
"""

import json
import os
import shutil
import unittest

from peep.src.processor_modules import (
    ImageProcessor,
    NDVICalculator,
    NetworkCentralityCalculator,
    WeatherImageToJSON,
)


@unittest.skipIf(
    os.environ.get("CI") == "true",
    "Skipping this test in a CI - waiting on a fix for #20. "
    "https://github.com/urbangrammarai/gee_pipeline/issues/20",
)
def test_Sentinel2_image_processor():
    """
    Should combine tif files into RGB, NDVI, and BWNDVI
    big images, and split RGB and BWNVI into sub-images.
    """
    dir_path = os.path.join(
        os.path.dirname(__file__), "..", "testdata", "Sentinel2", "test_tif"
    )
    tmp_png_path = os.path.join(
        os.path.dirname(__file__), "..", "testdata", "Sentinel2", "tmp_png"
    )

    vip = ImageProcessor()
    vip.input_location = dir_path
    vip.output_location = tmp_png_path
    vip.ndvi = True
    vip.bounds = [-3.0183, 53.3649, -2.9482, 53.4350]
    vip.configure()
    vip.run()
    assert os.path.exists(os.path.join(tmp_png_path, "2018-03-01", "PROCESSED"))
    assert len(os.listdir(os.path.join(tmp_png_path, "2018-03-01", "PROCESSED"))) == 3
    assert os.path.exists(os.path.join(tmp_png_path, "2018-03-01", "SPLIT"))
    assert len(os.listdir(os.path.join(tmp_png_path, "2018-03-01", "SPLIT"))) == 3468
    shutil.rmtree(tmp_png_path, ignore_errors=True)


def test_ERA5_image_to_json():
    """
    Get values out of tif files and put into JSON file.
    """
    dir_path = os.path.join(
        os.path.dirname(__file__), "..", "testdata", "ERA5", "test_tif"
    )
    tmp_json_path = os.path.join(
        os.path.dirname(__file__), "..", "testdata", "ERA5", "tmp_json"
    )

    wip = WeatherImageToJSON()
    wip.input_location = dir_path
    wip.output_location = tmp_json_path
    wip.bounds = [-3.0183, 53.3649, -2.9482, 53.4350]
    wip.configure()
    wip.run()
    assert os.path.exists(
        os.path.join(
            tmp_json_path, "2016-01-16", "JSON", "WEATHER", "weather_data.json"
        )
    )
    results = json.load(
        open(
            os.path.join(
                tmp_json_path, "2016-01-16", "JSON", "WEATHER", "weather_data.json"
            )
        )
    )
    #    assert "2016-01-16" in results.keys()
    assert "mean_2m_air_temperature" in results.keys()
    assert "total_precipitation" in results.keys()
    assert isinstance(results["mean_2m_air_temperature"], float)
    assert isinstance(results["total_precipitation"], float)
    shutil.rmtree(tmp_json_path)


def test_NDVICalculator():
    """
    Test that we can go from a directory containing some 50x50 BWNVI images
    to a json file containing network centrality values.
    """
    dir_path = os.path.join(
        os.path.dirname(__file__), "..", "testdata", "Sentinel2", "test_png"
    )
    tmp_json_path = os.path.join(
        os.path.dirname(__file__), "..", "testdata", "Sentinel2", "tmp_json"
    )
    ndvic = NDVICalculator()
    ndvic.input_location = dir_path
    ndvic.output_location = tmp_json_path
    ndvic.configure()
    ndvic.run()
    assert os.path.exists(
        os.path.join(tmp_json_path, "2018-03-01", "JSON", "NDVI", "ndvi_values.json")
    )
    nc_json = json.load(
        open(
            os.path.join(
                tmp_json_path, "2018-03-01", "JSON", "NDVI", "ndvi_values.json"
            )
        )
    )
    assert isinstance(nc_json, list)
    assert isinstance(nc_json[0], dict)
    # test float values
    for key in ["latitude", "longitude", "ndvi", "ndvi_veg"]:
        assert key in nc_json[0].keys()
        assert isinstance(nc_json[0][key], float)
        assert nc_json[0][key] != 0.0
    assert "date" in nc_json[0].keys()
    assert isinstance(nc_json[0]["date"], str)
    shutil.rmtree(tmp_json_path)
