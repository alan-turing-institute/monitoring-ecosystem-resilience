"""
Modules that can consolidate inputs from different sources
and produce combined output file (typically JSON).
"""
import os
import json

from file_utils import save_json

from pyveg_sequence import BaseModule

class VegAndWeatherJsonCombiner(BaseModule):
    """
    Expect directory structures like:
    <something>/<input_veg_dir>/<date>/network_centralities.json
    <something>/<input_weather_dir>/RESULTS/weather_data.json
    """

    def __init__(self, name):
        super().__init__(name)
        self.params += [
            ("output_dir", str),
            ("input_veg_dir", str),
            ("input_weather_dir", str),
            ("weather_collection", str),
            ("veg_collection", str)
            ]
        self.set_default_parameters()

    def set_default_parameters(self):
        if not "weather_collection" in vars(self):
            self.weather_collection = "ECMWF/ERA5/MONTHLY"
        if not "veg_collection" in vars(self):
            self.veg_collection = "COPERNICUS/S2"

    def get_veg_time_series(self):
        date_strings = os.listdir(self.input_veg_dir)
        veg_time_series = {}
        for date_string in date_strings:
            veg_json = os.path.join(self.input_veg_dir, date_string, "network_centralities.json")
            if not os.path.exists(veg_json):
                print("{}: no network centralities found for {}".format(
                    self.name, date_string))
                continue
            veg_time_point = json.load(open(veg_json))
            veg_time_series[date_string] = veg_time_point
        return veg_time_series


    def get_weather_time_series(self):
        weather_json = os.path.join(self.input_weather_dir,
                                    "RESULTS","weather_data.json")
        weather_time_series = json.load(open(weather_json))
        return weather_time_series


    def run(self):
        self.check_config()
        output_dict = {}

        weather_time_series = self.get_weather_time_series()
        output_dict[self.weather_collection] = {"type": "weather",
                                                "time-series-data": weather_time_series}
        veg_time_series = self.get_veg_time_series()
        output_dict[self.veg_collection] = {"type":"vegetation",
                                            "time-series-data": veg_time_series}
        save_json(output_dict, self.output_dir, "results_summary.json")
