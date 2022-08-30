from peep.configs.collections import data_collections
from peep.coordinates import coordinate_store

name = "birmingham"

# Define location to save all outputs
output_location = "birmingham-test"

#output_location_type = "azure"
output_location_type = "local"

# parse selection. Note (long, lat) GEE convention.
#entry = coordinate_store.loc[coordinate_id]
# modify this line to set coords based on entries in `coordinates.py`
#coordinate_id = "00"

# parse selection. Note (long, lat) GEE convention.
#entry = coordinate_store.loc[coordinate_id]
#coordinates = (entry.longitude, entry.latitude)

bounds = [-1.998575, 52.419471, -1.898575, 52.519471]

date_range = ["2020-05-01", "2020-08-01"]

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
    "Sentinel2": {"time_per_point": "3m"}     # this is a whole Sequence
    # and another Module
}
