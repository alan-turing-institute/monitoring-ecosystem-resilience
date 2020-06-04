"""
Build and run a pyveg pipeline based on a configuration json file.
"""

import os
import sys
import time
import json
import argparse
import inspect
import re
import datetime

from pyveg.src.date_utils import get_date_range_for_collection
from pyveg.src.pyveg_pipeline import Pipeline, Sequence
from pyveg.src.download_modules import VegetationDownloader, WeatherDownloader
from pyveg.src.processor_modules import (
    VegetationImageProcessor,
    NetworkCentralityCalculator,
    NDVICalculator,
    WeatherImageToJSON
)

from pyveg.src.combiner_modules import VegAndWeatherJsonCombiner



def build_module(module_class, config_file):
    """
    Load json config and instantiate modules
    """
    for n, c in inspect.getmembers(sys.modules[__name__]):
        if n == module_class:
            module = c()

    config_dict = json.load(open(config_file))
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
    parser.add_argument("--module_class", help="Class of module to instantiate", required=True)

    args = parser.parse_args()
    module = build_module(args.module_class, args.config_file)
    configure_and_run_module(module)


if __name__ == "__main__":
    main()
