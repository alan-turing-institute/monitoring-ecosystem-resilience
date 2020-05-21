
from pyveg.coordinates import coordinate_store
from pyveg.configs.collections import data_collections


#Define location to save all outputs
output_location = 'aztestoutput'
output_location_type = "azure"


# modify this line to set coords based on entries in `coordinates.py`
coordinate_id = '00'

# parse selection. Note (long, lat) GEE convention.
entry = coordinate_store.loc[coordinate_id]
coordinates = (entry.longitude, entry.latitude)

date_range = ['2016-01-01', '2016-02-01']


collections_to_use = ['Sentinel2', 'ERA5']

modules_to_use = {
    "ERA5": [
        "WeatherDownloader",
        "WeatherImageToJSON"
    ],
    "Sentinel2": [
        "VegetationDownloader",
        "VegetationImageProcessor",
        "NetworkCentralityCalculator"
    ],
    "combine": [
        "VegAndWeatherJsonCombiner"
        ]
}
