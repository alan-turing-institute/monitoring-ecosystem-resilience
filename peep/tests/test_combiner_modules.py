"""
Test the module that combines input from different collections
e.g. vegetation images and weather, to produce one output json file.
"""

import json
import os
import shutil
import unittest

from peep.src.combiner_modules import VegAndWeatherJsonCombiner


def test_instantiate_combiner():
    vwc = VegAndWeatherJsonCombiner()
    assert VegAndWeatherJsonCombiner


@unittest.skipIf(
    os.environ.get("CI") == "true",
    "Skipping this test in a CI - waiting on a fix for #19. "
    "https://github.com/urbangrammarai/gee_pipeline/issues/19",
)
def test_combine():
    input_veg_dir = os.path.join(
        os.path.dirname(__file__), "..", "testdata", "Sentinel2", "test_json"
    )
    input_weather_dir = os.path.join(
        os.path.dirname(__file__), "..", "testdata", "ERA5", "test_json"
    )
    output_dir = os.path.join(os.path.dirname(__file__), "..", "testdata", "tmp_json")
    vwc = VegAndWeatherJsonCombiner()
    vwc.input_veg_location = input_veg_dir
    vwc.input_weather_location = input_weather_dir
    vwc.output_location = output_dir
    vwc.veg_collection = "COPERNICUS/S2"
    vwc.weather_collection = "ECMWF/ERA5/MONTHLY"
    vwc.configure()
    vwc.run()
    assert os.path.exists(os.path.join(output_dir, "results_summary.json"))
    results = json.load(open(os.path.join(output_dir, "results_summary.json")))
    for coll in ["COPERNICUS/S2", "ECMWF/ERA5/MONTHLY"]:
        assert coll in results.keys()
        assert isinstance(results[coll], dict)
        assert "time-series-data" in results[coll].keys()
    assert "metadata" in results.keys()
    assert "tag" in results["metadata"].keys()
    assert len(results["metadata"]["tag"]) > 0
    shutil.rmtree(output_dir)
