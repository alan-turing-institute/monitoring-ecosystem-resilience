"""
Build and run a pyveg pipeline based on a configuration json file.
"""

import os
import sys
import time
import json
import argparse
import importlib.util
import inspect
from shutil import copyfile
import re
import datetime

import ee

from pyveg.src.date_utils import get_date_range_for_collection
from pyveg.src.pyveg_pipeline import Pipeline, Sequence

try:
    from pyveg.src.download_modules import VegetationDownloader, WeatherDownloader
except (ee.ee_exception.EEException):
    print("Earth Engine not initialized - will not be able to download from GEE")
    pass
from pyveg.src.processor_modules import (
    VegetationImageProcessor,
    NetworkCentralityCalculator,
    NDVICalculator,
    WeatherImageToJSON,
)

from pyveg.src.combiner_modules import VegAndWeatherJsonCombiner


def build_pipeline(config_file, from_cache=False):
    """
    Load json config and instantiate modules
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError("Unable to find config file {}".format(config_file))

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    spec = importlib.util.spec_from_file_location("myconfig", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # instantiate and setup the pipeline
    p = Pipeline(config.name)
    p.config_filename = os.path.basename(config_file)
    p.output_location = config.output_location
    if not from_cache:
        p.output_location += "__" + current_time
    else:
        # use the time from the filename
        time_match = re.search(
            "([\d]{4}-[\d]{2}-[\d]{2}_[\d]{2}-[\d]{2}-[\d]{2})",
            os.path.basename(config_file),
        )
        if time_match:
            p.output_location += "__" + time_match.groups()[0]
        else:
            print(
                "Wasn't able to infer timestamp from config filename.",
                "Will use original output_location from {}.".format(config_file),
            )
    p.output_location_type = config.output_location_type
    p.coords = config.coordinates
    p.date_range = config.date_range
    # if an id of a row in coordinates.py has been specified, add it here
    if "coords_id" in vars(config):
        p.coords_id = config.coords_id
    # if we have a pattern_type description, add it to the pipeline
    if "pattern_type" in vars(config):
        p.pattern_type = config.pattern_type
    if not from_cache:
        # before we run anything, save the current config to the configs dir
        config_cache_dir = os.path.join(os.path.dirname(config_file), "cached_config")
        os.makedirs(config_cache_dir, exist_ok=True)
        cached_config_file = (
            os.path.basename(config_file)[:-3] + "__" + current_time + ".py"
        )

        copyfile(config_file, os.path.join(config_cache_dir, cached_config_file))

    # add sequences to the pipeline to deal with different data types
    for coll in config.collections_to_use:
        s = Sequence(coll)
        coll_dict = config.data_collections[coll]
        s.set_config(coll_dict)
        # overwrite the date range with one that takes into account
        # the limits of this collection
        s.date_range = get_date_range_for_collection(config.date_range, coll_dict)

        # see if there's any special config for this sequence
        if "special_config" in vars(config):
            if coll in config.special_config.keys():
                s.set_config(config.special_config[coll])

                # add modules to the sequence
        for module_name in config.modules_to_use[coll]:
            for n, c in inspect.getmembers(sys.modules[__name__]):
                if n == module_name:
                    module = c()
                    if "special_config" in vars(config):
                        if n in config.special_config.keys():
                            module.set_parameters(config.special_config[n])
                    s += module
        # add the sequence to the pipeline
        p += s
    if len(config.collections_to_use) > 1:
        # now add the combiner module in its own sequence
        s = Sequence("combine")
        # Combiner needs the previous sequences to finish (in case we ever try to
        # parallelize further)
        s.depends_on = config.collections_to_use

        for module_name in config.modules_to_use["combine"]:
            for n, c in inspect.getmembers(sys.modules[__name__]):
                if n == module_name:
                    s += c()

        # and add this combiner sequence to the pipeline.
        p += s
    return p


def configure_and_run_pipeline(pipeline):
    """
    Call configure() run() on all sequences in the pipeline.
    """
    pipeline.configure()
    pipeline.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Path to config file", required=True)
    parser.add_argument(
        "--from_cache",
        help="Are we using a cached config file to resume an unfinished job?",
        action="store_true",
    )

    args = parser.parse_args()
    pipeline = build_pipeline(args.config_file, args.from_cache)
    configure_and_run_pipeline(pipeline)


if __name__ == "__main__":
    main()
