"""
Modules that can consolidate inputs from different sources
and produce combined output file (typically JSON).
"""
import os
import json

from pyveg.src.file_utils import save_json

from pyveg.src.pyveg_pipeline import BaseModule

class VegAndWeatherJsonCombiner(BaseModule):
    """
    Expect directory structures like:
    <something>/<input_veg_location>/<date>/network_centralities.json
    <something>/<input_weather_location>/RESULTS/weather_data.json
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.params += [
            ("output_location", [str]),
            ("input_veg_location", [str]),
            ("input_weather_location", [str]),
            ("output_location_type", [str]),
            ("input_veg_location_type", [str]),
            ("input_weather_location_type", [str]),
            ("weather_collection", [str]),
            ("veg_collection", [str])
            ]


    def set_default_parameters(self):
        # see if we can set our input directories from the output directories
        # of previous series in the pipeline.
        # The pipeline (if there is one) will be a grandparent, i.e. self.parent.parent
        super().set_default_parameters()
        if self.parent and self.parent.parent and self.parent.depends_on:
            for sequence_name in self.parent.depends_on:
                sequence = self.parent.parent.get(sequence_name)
                if sequence.data_type == "vegetation":
                    self.input_veg_location = sequence.output_location
                    self.input_veg_location_type = sequence.output_location_type

                    self.veg_collection = sequence.collection_name
                elif sequence.data_type == "weather":
                    self.input_weather_location = sequence.output_location
                    self.input_weather_location_type = sequence.output_location_type
                    self.weather_collection = sequence.collection_name
        else:
            self.weather_collection = "ECMWF/ERA5/MONTHLY"
            self.veg_collection = "COPERNICUS/S2"
            self.input_veg_location_type = "local"
            self.input_weather_location_type = "local"
            self.output_location_type = "local"


    def get_veg_time_series(self):
        date_strings = os.listdir(self.input_veg_location)
        date_strings.sort()
        veg_time_series = {}
        for date_string in date_strings:
            veg_json = os.path.join(self.input_veg_location, date_string, "network_centralities.json")
            if not os.path.exists(veg_json):
                print("{}: no network centralities found for {}".format(
                    self.name, date_string))
                continue
            veg_time_point = json.load(open(veg_json))
            veg_time_series[date_string] = veg_time_point
        return veg_time_series


    def get_weather_time_series(self):
        weather_json = os.path.join(self.input_weather_location,
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
        save_json(output_dict, self.output_location, "results_summary.json")
