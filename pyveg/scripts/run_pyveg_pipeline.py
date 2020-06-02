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

from pyveg.src.pyveg_pipeline import Pipeline, Sequence
from pyveg.src.download_modules import VegetationDownloader, WeatherDownloader
from pyveg.src.processor_modules import (
    VegetationImageProcessor,
    NetworkCentralityCalculator,
    NDVICalculator,
    WeatherImageToJSON
)

from pyveg.src.combiner_modules import VegAndWeatherJsonCombiner


def build_pipeline(config_file):
    """
    Load json config and instantiate modules
    """
    spec = importlib.util.spec_from_file_location("myconfig", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # instantiate and setup the pipeline
    p = Pipeline(config.name)
    p.output_location = config.output_location
    p.output_location += '__' + time.strftime("%Y-%m-%d_%H-%M-%S")
    p.output_location_type = config.output_location_type
    p.coords = config.coordinates
    p.date_range = config.date_range

    # before we run anything, save the current config to the configs dir
    configs_dir = os.path.dirname(config_file)
    cached_config_file = config_file[:-3] + \
        '__' + time.strftime("%Y-%m-%d_%H-%M-%S") + ".py"

    copyfile(config_file, cached_config_file)

    if config.output_location_type=="local" and not os.path.exists(p.output_location):
        os.makedirs(p.output_location, exist_ok=True)
    # add sequences to the pipeline to deal with different data types
    for coll in config.collections_to_use:
        s = Sequence(coll)
        coll_dict = config.data_collections[coll]
        s.set_config(coll_dict)
        # add modules to the sequence
        for module_name in config.modules_to_use[coll]:
            for n, c in inspect.getmembers(sys.modules[__name__]):
                if n == module_name:
                    module = c()
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

    args = parser.parse_args()
    pipeline = build_pipeline(args.config_file)
    configure_and_run_pipeline(pipeline)


if __name__ == "__main__":
    main()
