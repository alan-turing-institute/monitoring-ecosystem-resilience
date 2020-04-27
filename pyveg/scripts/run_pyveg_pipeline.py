"""
Build and run a pyveg pipeline based on a configuration json file.
"""

import os
import json
import argparse
import importlib.util

from pyveg.src.pyveg_pipeline import Pipeline, Sequence
from pyveg.src.download_modules import VegetationDownloader, WeatherDownloader
from pyveg.src.processor_modules import (
    VegetationImageProcessor,
    NetworkCentralityCalculator,
    WeatherImageToJSON
)

from pyveg.src.combiner_modules import VegAndWeatherJsonCombiner


def build_pipeline(config_file, name="mypyveg"):
    """
    Load json config and instantiate modules
    """
    spec = importlib.util.spec_from_file_location("myconfig",config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    # instantiate and setup the pipeline
    p = Pipeline(name)
    p.output_dir = config.output_dir
    p.coords = config.coordinates
    p.date_range = config.date_range

    # add sequences to the pipeline to deal with different data types
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

    # add this combiner module to the combiner sequence
    s += cm
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
    parser.add_argument("--name", help="(optional) identifying name", default="mypyveg")

    args = parser.parse_args()
    pipeline = build_pipeline(args.config_file, args.name)
    configure_and_run_pipeline(pipeline)


if __name__ == "__main__":
    main()
