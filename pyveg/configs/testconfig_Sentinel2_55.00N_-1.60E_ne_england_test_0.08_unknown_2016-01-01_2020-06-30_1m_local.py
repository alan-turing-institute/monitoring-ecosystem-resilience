# config file for running pyveg_run_pipeline
##
# This file was generated from the command
# pyveg_generate_config
# on 22-07-15 15:01:54

from pyveg.configs.collections import data_collections

# this name doesn't do anything - it's just an id given to the Pipeline instance
name = "pyveg"

# Define location to save all outputs.   Note that every time the pipeline job
# is rerun, a datestamp will be appended to the output_location.
output_location = "./Sentinel2-55.00N-1.60W-ne_england_test"
output_location_type = "local"

# parse selection. Note (long, lat) GEE convention.
coordinates = (-1.60, 55.00)

# optional coords_id setting


# pattern_type description
pattern_type = "unknown"

date_range = ["2016-01-01", "2020-06-30"]

# From the dictionary entries in data_collections.py, which shall we use
# (these will become "Sequences")
collections_to_use = ["Sentinel2", "ERA5"]

# Dictionary defining what Modules should run in each Sequence.

modules_to_use = {
    "ERA5": ["WeatherDownloader", "WeatherImageToJSON"],
    "Sentinel2": [
        "VegetationDownloader",
        "VegetationImageProcessor",
        "NetworkCentralityCalculator",
        "NDVICalculator",
    ],
    "combine": ["VegAndWeatherJsonCombiner"],
}

# The following demonstrates how parameters can be set for individual Modules:
special_config = {
    "Sentinel2": {"time_per_point": "1m"},
    "ERA5": {"time_per_point": "1m", "date_range": ["2016-01-01", "2020-06-30"]},
    "VegetationDownloader": {"region_size": 0.08},
    "VegetationImageProcessor": {"run_mode": "local"},
    "NetworkCentralityCalculator": {
        "n_threads": 4,
        "run_mode": "local",
        "n_sub_images": 10,
    },
    "NDVICalculator": {"run_mode": "local", "n_sub_images": 10},
}
