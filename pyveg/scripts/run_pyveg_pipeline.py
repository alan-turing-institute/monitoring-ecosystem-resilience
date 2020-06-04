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



def build_pipeline(config_file, from_cache=False):
    """
    Load json config and instantiate modules
    """

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    spec = importlib.util.spec_from_file_location("myconfig", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # instantiate and setup the pipeline
    p = Pipeline(config.name)
    p.output_location = config.output_location
    if not from_cache:
        p.output_location += '__' + current_time
    else:
        # use the time from the filename
        time_match = re.search("([\d]{4}-[\d]{2}-[\d]{2}_[\d]{2}-[\d]{2}-[\d]{2})",
                               os.path.basename(config_file))
        if time_match:
            p.output_location += '__' + time_match.groups()[0]
        else:
            print("Wasn't able to infer timestamp from config filename.",
                  "Will use original output_location from {}.".format(config_file))
    p.output_location_type = config.output_location_type
    p.coords = config.coordinates
    p.date_range = config.date_range
    if not from_cache:
        # before we run anything, save the current config to the configs dir
        config_cache_dir = os.path.join(os.path.dirname(config_file),"cached_config")
        os.makedirs(config_cache_dir, exist_ok=True)
        cached_config_file = os.path.basename(config_file)[:-3] + \
            '__' + current_time + ".py"

        copyfile(config_file, os.path.join(config_cache_dir, cached_config_file))

    if config.output_location_type=="local" and not os.path.exists(p.output_location):
        os.makedirs(p.output_location, exist_ok=True)
    # add sequences to the pipeline to deal with different data types
    for coll in config.collections_to_use:
        s = Sequence(coll)
        coll_dict = config.data_collections[coll]
        s.set_config(coll_dict)
        # overwrite the date range with one that takes into account
        # the limits of this collection
        s.date_range = get_date_range_for_collection(config.date_range, coll_dict)
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
    parser.add_argument("--from_cache", help="Are we using a cached config file to resume an unfinished job?",
                        action='store_true')

    args = parser.parse_args()
    pipeline = build_pipeline(args.config_file, args.from_cache)
    configure_and_run_pipeline(pipeline)


if __name__ == "__main__":
    main()
