from peep.configs.collections import data_collections
from peep.coordinates import coordinate_store

# Name of job
name = "london"

# Define location to save all outputs
output_location = "london-test"
output_location_type = "local"

bounds = [-0.0509564, 51.4425565, 0.0509564, 51.5225565]


date_range = ["2017-05-01", "2017-09-01"]

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
special_config = {
    "Sentinel2": {"time_per_point": "3m"}
}
