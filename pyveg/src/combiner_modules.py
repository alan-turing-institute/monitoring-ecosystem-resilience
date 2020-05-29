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

    def combine_json_lists(self, json_lists):
        print("Will combine {} json lists".format(len(json_lists)))
        if len(json_lists) == 1:
            return json_lists[0]
        ## any way to do this without a huge nested loop?

        # loop over all the lists apart from the first, which we will add to
        for jlist in json_lists[1:]:
            # loop through all items (sub-images) in each list
            for p in jlist:
                match_found = False
                # loop through all items (sub-images) in the first/output list
                for p0 in json_lists[0]:
                    # match by latitude, longitude.
                    if (p["latitude"],p["longitude"],p["date"]) == \
                       (p0["latitude"],p0["longitude"],p0["date"]):
                        match_found = True
                        for k,v in p.items():
                            if not k in p0.keys():
                                p0[k] = v
                        break
                if not match_found:
                    json_lists[0].append(p)
        return json_lists[0]



    def get_veg_time_series(self):
        date_strings = self.list_directory(self.input_veg_location,
                                           self.input_veg_location_type)
        date_strings.sort()
        veg_time_series = {}
        for date_string in date_strings:
            subdirs = self.list_directory(os.path.join(self.input_veg_location, date_string,"JSON"),
                                                       self.input_veg_location_type)
            veg_lists = []
            for subdir in subdirs:
                print("looking at {}".format(os.path.join(self.input_veg_location,
                                                          date_string,"JSON",
                                                   subdir)))
                dir_contents = self.list_directory(
                    os.path.join(
                        self.input_veg_location, date_string, "JSON", subdir),
                    self.input_veg_location_type)
                print("Dir contents are {}".format(dir_contents))
                json_files = [filename for filename in dir_contents if filename.endswith(".json")]
                for filename in json_files:
                    j = self.get_json(os.path.join(self.input_veg_location,
                                                   date_string,
                                                   "JSON",
                                                   subdir,
                                                   filename),
                                      self.input_veg_location_type)
                    veg_lists.append(j)

            veg_time_point = self.combine_json_lists(veg_lists)

            veg_time_series[date_string] = veg_time_point
        return veg_time_series


    def get_weather_time_series(self):
        date_strings = self.list_directory(self.input_weather_location,
                                           self.input_weather_location_type)
        date_strings.sort()
        weather_time_series = {}
        for date_string in date_strings:

            weather_json = self.get_json(os.path.join(self.input_weather_location,
                                                      date_string,"JSON","WEATHER",
                                                      "weather_data.json"),
                                         self.input_weather_location_type)
            weather_time_series[date_string] = weather_json
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
        self.save_json(output_dict, "results_summary.json", self.output_location,
                       self.output_location_type)
