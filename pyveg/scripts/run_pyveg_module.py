"""
Build and run a pyveg pipeline based on a configuration json file.
"""
import argparse
import datetime
import inspect
import json
import os
import re
import sys
import time

import ee

from pyveg.src.date_utils import get_date_range_for_collection
from pyveg.src.pyveg_pipeline import Pipeline, Sequence

try:
    from pyveg.src.download_modules import VegetationDownloader, WeatherDownloader
except (ee.ee_exception.EEException):
    print("Earth Engine not initialized - will not be able to download from GEE")
    pass
from pyveg.src.combiner_modules import VegAndWeatherJsonCombiner
from pyveg.src.processor_modules import (
    NDVICalculator,
    NetworkCentralityCalculator,
    VegetationImageProcessor,
    WeatherImageToJSON,
)


def build_module(config_file):
    """
    Load json config and instantiate modules
    """
    config_dict = json.load(open(config_file))
    if not "class_name" in config_dict.keys():
        raise RuntimeError("Need to have class_name defined in the config.")
    module_class = config_dict["class_name"]
    for n, c in inspect.getmembers(sys.modules[__name__]):
        if n == module_class:
            module = c()

    module.set_parameters(config_dict)
    return module


def configure_and_run_module(module):
    """
    Call configure() run() on all the module
    """
    module.configure()
    module.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Path to config file", required=True)

    args = parser.parse_args()
    module = build_module(args.config_file)
    configure_and_run_module(module)


if __name__ == "__main__":
    main()
