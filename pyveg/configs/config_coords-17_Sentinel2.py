from pyveg.coordinates import coordinate_store
from pyveg.configs.collections import data_collections

name = "pyvegtest"

# Define location to save all outputs
output_location = "coords17-Sentinel2"
output_location_type = "azure"
# output_location_type = "local"


# modify this line to set coords based on entries in `coordinates.py`
coordinate_id = "17"

# parse selection. Note (long, lat) GEE convention.
entry = coordinate_store.loc[coordinate_id]
coordinates = (entry.longitude, entry.latitude)

date_range = ["2016-01-01", "2020-04-01"]

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
special_config = {"NetworkCentralityCalculator": {"n_sub_images": -1, "n_threads": 8}}
