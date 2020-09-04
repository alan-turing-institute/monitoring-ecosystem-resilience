##############  config file for running pyveg_run_pipeline
##
## This file was generated from the command
## pyveg_generate_config
## on CURRENT_TIME

from pyveg.configs.collections import data_collections

# this name doesn't do anything - it's just an id given to the Pipeline instance
name = "pyveg"

# Define location to save all outputs.   Note that every time the pipeline job
# is rerun, a datestamp will be appended to the output_location.
output_location = "OUTPUT_LOCATION"
output_location_type = "OUTPUT_LOCATION_TYPE"

# parse selection. Note (long, lat) GEE convention.
coordinates = (LONGITUDE, LATITUDE)

date_range = ["START_DATE", "END_DATE"]

# From the dictionary entries in data_collections.py, which shall we use
# (these will become "Sequences")
collections_to_use = ["COLLECTION_NAME", "WEATHER_COLL_NAME"]

# Dictionary defining what Modules should run in each Sequence.

modules_to_use = {
    "WEATHER_COLL_NAME": ["WeatherDownloader", "WeatherImageToJSON"],
    "COLLECTION_NAME": [
        "VegetationDownloader",
        "VegetationImageProcessor",
        "NetworkCentralityCalculator",
        "NDVICalculator",
    ],
    "combine": ["VegAndWeatherJsonCombiner"],
}

# The following demonstrates how parameters can be set for individual Modules:
special_config = {
    "COLLECTION_NAME": {
        "time_per_point": "TIME_PER_POINT"
    },
    "WEATHER_COLL_NAME": {
        "time_per_point": "TIME_PER_POINT",
        "date_range": ["WEATHER_STARTDATE", "END_DATE"]
    },
    "VegetationImageProcessor": {"run_mode": "RUN_MODE"},
    "NetworkCentralityCalculator": {
        "n_threads": NUM_THREADS,
        "run_mode": "RUN_MODE",
        "n_sub_images": NUM_SUBIMAGES
    },
    "NDVICalculator": {
        "run_mode": "RUN_MODE",
        "n_sub_images": NUM_SUBIMAGES
    }
}
