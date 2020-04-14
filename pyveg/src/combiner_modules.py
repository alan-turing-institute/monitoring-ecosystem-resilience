"""
Modules that can consolidate inputs from different sources
and produce combined output file (typically JSON).
"""

import json

from pyveg_sequence import AnalysisModule

class VegAndWeatherJsonCombiner(AnalysisModule):
    """
    Expect directory structures like:
    <something>/<input_veg_dir>/<date>/network_centralities.json
    <something>/<input_weather_dir>/RESULTS/weather_data.json
    """

    def __init__(self, name):
        super().__init__(name)
        self.parameters += [
            ("output_dir", str),
            ("input_veg_dir", str),
            ("input_weather_dir", str),
            ("weather_collection", str),
            ("veg_collection", str)
            ]

    def set_default_parameters(self):
        if not "weather_collection" in vars(self):
            self.weather_collection = "ECMWF-ERA5-MONTHLY"
        if not "veg_collection" in vars(self):
            self.veg_collection = "COPERNICUS/S2"


    def run(self):
        output_dict = {}
