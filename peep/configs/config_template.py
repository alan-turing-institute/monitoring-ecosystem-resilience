# config file for running pyveg_run_pipeline
##
# This file was generated from the command
# pyveg_generate_config
# on CURRENT_TIME

from peep.configs.collections import data_collections

# this name doesn't do anything - it's just an id given to the Pipeline instance
name = "peep"

# Define location to save all outputs.   Note that every time the pipeline job
# is rerun, a datestamp will be appended to the output_location.
output_location = "OUTPUT_LOCATION"
output_location_type = "local"

# parse selection. Note (long, lat) GEE convention.
bounds = [LEFT, BOTTOM, RIGHT, TOP]

date_range = ["START_DATE", "END_DATE"]

# From the dictionary entries in data_collections.py, which shall we use
# (these will become "Sequences")
collections_to_use = ["COLLECTION_NAME"]

# Dictionary defining what Modules should run in each Sequence.

modules_to_use = {
    "COLLECTION_NAME": [
        "VegetationDownloader",
        "VegetationImageProcessor",
    ],
}

# The following demonstrates how parameters can be set for individual Modules:
special_config = {
    "COLLECTION_NAME": {
        "time_per_point": "TIME_PER_POINT"
    },
}
