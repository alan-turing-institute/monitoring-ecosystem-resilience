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
        s.set_config(coll_dict)
        if coll_dict["data_type"] == "vegetation":
            # add a vegetation downloader, a vegetation image_processor,
            # and a network centrality calculator.
            s += VegetationDownloader()
            s += VegetationImageProcessor()
            s += NetworkCentralityCalculator()

        elif coll_dict["data_type"] == "weather":
            # add a downloader module and a module to convert to json.
            s += WeatherDownloader()
            s += WeatherImageToJSON()
        # add the sequence to the pipeline
        p += s
    # now add the combiner module in its own sequence
    s = Sequence("combine")
    # Combiner needs the previous sequences to finish (in case we ever try to
    # parallelize further)
    s.depends_on = config.collections_to_use
    cm = VegAndWeatherJsonCombiner()
 #   for sequence in p.sequences:
 #       if "data_type" in vars(sequence):
 #           if sequence.data_type == "vegetation":
 #               print("SETTING VEG STUFF FOR COMBINER {} {}".format(sequence.output_dir,
 #                                                                   sequence.collection_name))
 #               cm.input_veg_dir = sequence.output_dir
 #               cm.veg_collection = sequence.collection_name
 #           elif sequence.data_type == "weather":
 #               cm.input_weather_dir = sequence.output_dir
 #               cm.weather_collection = sequence.collection_name
    # add this combiner module to the combiner sequence
    s += cm
    # and add this combiner sequence to the pipeline.
    p += s
    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Path to config file", required=True)
    parser.add_argument("--name", help="(optional) identifying name", default="mypyveg")

    args = parser.parse_args()
    pipeline = build_pipeline(args.config_file, args.name)
