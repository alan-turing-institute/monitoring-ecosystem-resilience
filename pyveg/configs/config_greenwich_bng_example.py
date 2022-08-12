from pyveg.configs.collections import data_collections
from pyveg.coordinates import coordinate_store

name = "greenwich"

# Define location to save all outputs
output_location = "greenwich-bng"
#output_location_type = "azure"
output_location_type = "local"

# parse selection. Note (long, lat) GEE convention.
#entry = coordinate_store.loc[coordinate_id]
# modify this line to set coords based on entries in `coordinates.py`
#coordinate_id = "00"

# parse selection. Note (long, lat) GEE convention.
#entry = coordinate_store.loc[coordinate_id]
#coordinates = (entry.longitude, entry.latitude)

# bounds = [ 327771, 384988, 339052, 395656 ]
# bounds = [297566, 345093, 387814, 430437]
bounds = [532480.0, 174080.0, 542720.0, 184320.0]
date_range = ["2016-05-01", "2016-08-01"]

# From the dictionary entries in data_collections.py, which shall we use
# (these will become "Sequences")
collections_to_use = ["Sentinel2"]

# Dictionary defining what Modules should run in each Sequence.

modules_to_use = {
    "Sentinel2": [
        "VegetationDownloader",
        "VegetationImageProcessor",
    ],
}

# The following demonstrates how parameters can be set for individual Modules or Sequences:
special_config = {"Sentinel2": {"time_per_point": "3m"}     # this is a whole Sequence
                  #    # and another Module
                  }
