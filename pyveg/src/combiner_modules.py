"""
Modules that can consolidate inputs from different sources
and produce combined output file (typically JSON).
"""
import os
import json

from pyveg.src.file_utils import save_json, get_tag
from pyveg.src.date_utils import get_date_strings_for_time_period
from pyveg.src.pyveg_pipeline import BaseModule, logger


class CombinerModule(BaseModule):
    def __init__(self, name=None):
        super().__init__(name)
        self.params += [("output_location", [str]), ("output_location_type", [str])]


class VegAndWeatherJsonCombiner(CombinerModule):
    """
    Expect directory structures like:
    <something>/<input_veg_location>/<date>/network_centralities.json
    <something>/<input_weather_location>/RESULTS/weather_data.json
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.params += [
            ("input_veg_location", [str]),
            ("input_weather_location", [str]),
            ("input_veg_location_type", [str]),
            ("input_weather_location_type", [str]),
            ("weather_collection", [str]),
            ("veg_collection", [str]),
            ("output_filename", [str]),
        ]


    def set_default_parameters(self):
        """
        See if we can set our input directories from the output directories
        of previous Sequences in the pipeline.
        The pipeline (if there is one) will be a grandparent,
        i.e. self.parent.parent
        and the names of the Sequences we will want to combine should be
        in the variable self.depends_on.
        """
        super().set_default_parameters()
        # get the parent Sequence and Pipeline
        if self.parent and self.parent.parent:
            # we're running in a Pipeline
            for seq_name in self.parent.depends_on:
                seq = self.parent.parent.get(seq_name)
                if seq.data_type == "vegetation":
                    self.input_veg_sequence = seq_name
                elif seq.data_type == "weather":
                    self.input_weather_sequence = seq_name
            if not (
                "input_veg_sequence" in vars(self)
                and "input_weather_sequence" in vars(self)
            ):
                raise RuntimeError(
                    "{}: Unable to find vegetation and weather sequences in depends_on".format(
                        self.name, self.depends_on
                    )
                )
            # now get other details from the input sequences
            veg_sequence = self.parent.parent.get(self.input_veg_sequence)
            self.input_veg_location = veg_sequence.output_location
            self.input_veg_location_type = veg_sequence.output_location_type
            self.veg_collection = veg_sequence.collection_name

            weather_sequence = self.parent.parent.get(self.input_weather_sequence)
            self.input_weather_location = weather_sequence.output_location
            self.input_weather_location_type = weather_sequence.output_location_type
            self.weather_collection = weather_sequence.collection_name
        else:
            # No parent Sequence or Pipeline - perhaps running standalone
            self.weather_collection = "ECMWF/ERA5/MONTHLY"
            self.veg_collection = "COPERNICUS/S2"
            self.input_veg_location_type = "local"
            self.input_weather_location_type = "local"
            self.output_location_type = "local"
        if not "output_filename" in vars(self):
            self.output_filename = "results_summary.json"

    def combine_json_lists(self, json_lists):
        """
        If for example we have json files from the NetworkCentrality
        and NDVI calculators, all containing lists of dicts for sub-images,
        combine them here by matching by coordinate.
        """
        if len(json_lists) == 0:
            return None
        elif len(json_lists) == 1:
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
                    if (p["latitude"], p["longitude"], p["date"]) == (
                        p0["latitude"],
                        p0["longitude"],
                        p0["date"],
                    ):
                        match_found = True
                        for k, v in p.items():
                            if not k in p0.keys():
                                p0[k] = v
                        break
                if not match_found:
                    json_lists[0].append(p)
        return json_lists[0]

    def get_veg_time_series(self):
        """
        Combine contents of JSON files written by the NetworkCentrality
        and NDVI calculator Modules.
        If we are running in a Pipeline, get the expected set of date strings
        from the vegetation sequence we depend on, and if there is no data
        for a particular date, make a null entry in the output.
        """

        dates_with_data = self.list_directory(
            self.input_veg_location, self.input_veg_location_type
        )

        if self.parent and self.parent.parent and "input_veg_sequence" in vars(self):
            veg_sequence = self.parent.parent.get(self.input_veg_sequence)
            start_date, end_date = veg_sequence.date_range
            time_per_point = veg_sequence.time_per_point
            date_strings = get_date_strings_for_time_period(
                start_date, end_date, time_per_point
            )
        else:
            date_strings = dates_with_data
        date_strings.sort()
        veg_time_series = {}
        for date_string in date_strings:
            if not date_string in dates_with_data:
                veg_time_series[date_string] = None
            # if there is no JSON directory for this date, add a null entry
            if "JSON" not in self.list_directory(
                self.join_path(self.input_veg_location, date_string),
                self.input_veg_location_type,
            ):
                veg_time_series[date_string] = None
                continue
            # find the subdirs of the JSON directory
            subdirs = self.list_directory(
                self.join_path(self.input_veg_location, date_string, "JSON"),
                self.input_veg_location_type,
            )
            veg_lists = []
            for subdir in subdirs:
                logger.debug(
                    "{}: getting vegetation time series for {}".format(
                        self.name,
                        self.join_path(
                            self.input_veg_location, date_string, "JSON", subdir
                        ),
                    )
                )
                # list the JSON subdirectories and find any .json files in them
                dir_contents = self.list_directory(
                    self.join_path(self.input_veg_location, date_string, "JSON", subdir),
                    self.input_veg_location_type,
                )
                json_files = [
                    filename for filename in dir_contents if filename.endswith(".json")
                ]
                for filename in json_files:
                    j = self.get_json(
                        self.join_path(
                            self.input_veg_location,
                            date_string,
                            "JSON",
                            subdir,
                            filename,
                        ),
                        self.input_veg_location_type,
                    )
                    veg_lists.append(j)
            # combine the lists from the different subdirectories
            veg_time_point = self.combine_json_lists(veg_lists)
            # update the final veg_time_series dictionary
            veg_time_series[date_string] = veg_time_point

        return veg_time_series

    def get_weather_time_series(self):
        date_strings = self.list_directory(
            self.input_weather_location, self.input_weather_location_type
        )
        date_strings.sort()
        weather_time_series = {}
        for date_string in date_strings:

            weather_json = self.get_json(
                self.join_path(
                    self.input_weather_location,
                    date_string,
                    "JSON",
                    "WEATHER",
                    "weather_data.json",
                ),
                self.input_weather_location_type,
            )
            weather_time_series[date_string] = weather_json
        return weather_time_series

    def check_output_dict(self, output_dict):
        """
        For all the keys  (i.e. dates) in the vegetation time-series,
        count how many have data for both veg and weather
        """
        veg_dates = output_dict[self.veg_collection]["time-series-data"].keys()
        weather_dates = output_dict[self.weather_collection]["time-series-data"].keys()
        for date in veg_dates:

            if output_dict[self.veg_collection]["time-series-data"][date] \
               and date in weather_dates \
               and output_dict[self.weather_collection]["time-series-data"][date]:
                self.run_status["succeeded"] += 1
        return

    def get_metadata(self):
        """
        Fill a dictionary with info about this job - coords, date range etc.
        """
        metadata = {}
        if self.parent and self.parent.parent and "input_veg_sequence" in vars(self):
            veg_sequence = self.parent.parent.get(self.input_veg_sequence)
            metadata["start_date"], metadata["end_date"] = veg_sequence.date_range
            metadata["time_per_point"] = veg_sequence.time_per_point
            metadata["longitude"] = veg_sequence.coords[0]
            metadata["latitude"] = veg_sequence.coords[1]
            metadata["collection"] = veg_sequence.collection_name
            metadata["num_data_points"] = self.run_status["succeeded"]
            if "config_filename" in vars(self.parent.parent):
                metadata["config_filename"] = self.parent.parent.config_filename
            if "coords_id" in vars(self.parent.parent):
                metadata["coords_id"] = self.parent.parent.coords_id
        metadata["tag"] = get_tag()
        return metadata

    def run(self):
        self.check_config()
        output_dict = {}
        logger.info("{}: getting weather time series".format(self.name))
        weather_time_series = self.get_weather_time_series()
        output_dict[self.weather_collection] = {
            "type": "weather",
            "time-series-data": weather_time_series,
        }
        logger.info("{}: getting vegetation time series".format(self.name))
        veg_time_series = self.get_veg_time_series()
        output_dict[self.veg_collection] = {
            "type": "vegetation",
            "time-series-data": veg_time_series,
        }
        logger.info("{}: checking combined JSON".format(self.name))
        self.check_output_dict(output_dict)
        logger.info("{}: filling metadata dict".format(self.name))
        metadata_dict = self.get_metadata()
        output_dict["metadata"] = metadata_dict
        self.save_json(
            output_dict,
            self.output_filename,
            self.output_location,
            self.output_location_type,
        )

        logger.info("{}: Wrote output to {}".format(
            self.name,
            self.join_path(self.output_location, self.output_filename)
        )
        )
        self.is_finished = True
