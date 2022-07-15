from pyveg.coordinates import coordinate_store
from pyveg.configs.collections import data_collections

name = "liverpool"

# Define location to save all outputs
output_location = "liverpool-test"
#output_location_type = "azure"
output_location_type = "local"

# parse selection. Note (long, lat) GEE convention.
#entry = coordinate_store.loc[coordinate_id]
# modify this line to set coords based on entries in `coordinates.py`
#coordinate_id = "00"

# parse selection. Note (long, lat) GEE convention.
#entry = coordinate_store.loc[coordinate_id]
#coordinates = (entry.longitude, entry.latitude)

coordinates = (-2.983333, 53.400002)

date_range = ["2016-05-01", "2021-08-01"]

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
special_config = {
    "Sentinel2": {"time_per_point": "3m", "region_size": 0.1125,'ndvi':True
}     # this is a whole Sequence
           # and another Module
}
