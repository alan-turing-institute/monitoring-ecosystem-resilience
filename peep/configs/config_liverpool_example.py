from peep.configs.collections import data_collections

name = "liverpool"

# Define location to save all outputs
output_location = "liverpool-test"
#output_location_type = "azure"
output_location_type = "local"

bounds = [-3.0183, 53.3649, -2.9482, 53.4350]

date_range = ["2016-05-01", "2016-08-01"]

# From the dictionary entries in data_collections.py, which shall we use
# (these will become "Sequences")
collections_to_use = ["Sentinel2"]

# Dictionary defining what Modules should run in each Sequence.

modules_to_use = {
    "Sentinel2": [
        "ImageDownloader",
        "ImageProcessor",
    ],
}

# The following demonstrates how parameters can be set for individual Modules or Sequences:
special_config = {"Sentinel2": {"time_per_point": "3m"}     # this is a whole Sequence
                  #    # and another Module
                  }
