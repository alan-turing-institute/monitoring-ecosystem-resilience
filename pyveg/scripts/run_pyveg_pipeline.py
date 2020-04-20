"""
Build and run a pyveg pipeline based on a configuration json file.
"""

import os
import json
import argparse
import importlib.util

from pyveg.src.pyveg_pipeline import Pipeline, Sequence
from pyveg.src.download_modules import *
from pyveg.src.analysis_modules import *
from pyveg.src.combiner_modules import *


def build_pipeline(config_file, name):
    """
    Load json config and instantiate modules
    """
    spec = importlib.util.spec_from_file_location("myconfig",config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    p = Pipeline(name)
    p.output_dir = config.output_dir
    p.coords = config.coordinates
    p.date_range = config.date_range
    for coll in config.collections_to_use:
        s = Sequence(coll)
        coll_dict = config.data_collections[coll]
        if coll_dict["type"] == "vegetation":
            # add a vegetation downloader, a vegetation image_processor,
            # and a network centrality calculator.
            s += VegetationDownloader()
            s += VegetationImageProcessor()
            s += NetworkCentralityCalculator()

        elif coll_dict["type"] == "weather":
            s += WeatherDownloader()
            s += WeatherImageToJSON()
        p += s
        return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Path to config file", required=True)
    parser.add_argument("--name", help="(optional) identifying name", default="test")

    args = parser.parse_args()
    pipeline = build_pipeline(args.config_file, args.name)
