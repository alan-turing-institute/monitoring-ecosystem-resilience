#!/usr/bin/env python

"""
Generate a config file pyveg/configs/<config_filename> for use when running
download and processing jobs with
pyveg_run_pipeline --config_file pyveg/configs/<config_filename>

User specifies:
* Coordinates  OR id of location in coordinates.py
* Date range
* time per point
* Satellite collection name (e.g. "Sentinel2", "Landsat8")
* run mode ("local" or "batch")
* whether to run in 'test' mode (fewer dates, and only a few sub-images).

These can be given directly as command-line arguments, or the user will
be prompted for them.

Usage
=====

pyveg_generate_config

then respond to prompts, or

pyveg_generate_config --help

to see a list of command line options.
(Note that command line options and prompted inputs can be mixed-and-matched).

"""

import argparse
import os
import re
import time

from pyveg.configs import collections
from pyveg.coordinates import coordinate_store
from pyveg.src.coordinate_utils import lookup_country


def get_template_text():
    template_filepath = os.path.join(
        os.path.dirname(__file__), "..", "configs", "config_template.py"
    )
    if not os.path.exists(template_filepath):
        raise RuntimeError("Unable to find template {}".format(template_filepath))
    return open(template_filepath).read()


def make_output_location(coords_id, collection_name, latitude, longitude, country):
    # quite restricted on characters allowed in Azure container names -
    # use NSEW rather than negative numbers in coordinates
    if latitude.startswith("-"):
        latitude = latitude[1:] + "S"
    else:
        latitude = latitude + "N"
    if longitude.startswith("-"):
        longitude = longitude[1:] + "W"
    else:
        longitude = longitude + "E"

    if coords_id:
        output_location = (
            f"{coords_id}-{collection_name}-{latitude}-{longitude}-{country}"
        )
    else:
        output_location = f"{collection_name}-{latitude}-{longitude}-{country}"
    return output_location


def make_filename(
    configs_dir,
    test_mode,
    longitude,
    latitude,
    country,
    pattern_type,
    start_date,
    end_date,
    time_per_point,
    region_size,
    collection_name,
    run_mode,
    coords_id,
):
    """
    Construct a filename from the specified parameters.
    """
    filename_start = "testconfig" if test_mode else "config"
    if coords_id:
        filename_start += "_" + coords_id
    filepath = os.path.join(
        configs_dir,
        f"{filename_start}_{collection_name}_{latitude}N_{longitude}E_{country}_{region_size}_{pattern_type}_{start_date}_{end_date}_{time_per_point}_{run_mode}.py",
    )
    return filepath


def write_file(
    configs_dir,
    output_location,
    longitude,
    latitude,
    country,
    pattern_type,
    start_date,
    end_date,
    time_per_point,
    region_size,
    collection_name,
    run_mode,
    n_threads,
    test_mode=False,
    coords_id=None,
):
    """
    Take the arguments, construct a filename, and write contents
    """
    filename = make_filename(
        configs_dir,
        test_mode,
        longitude,
        latitude,
        country,
        pattern_type,
        start_date,
        end_date,
        time_per_point,
        region_size,
        collection_name,
        run_mode,
        coords_id,
    )

    if time_per_point.endswith("d") or time_per_point.endswith("w"):
        weather_collection_name = "ERA5_daily"
        weather_start_date = start_date
    else:
        weather_collection_name = "ERA5"
        if test_mode:
            weather_start_date = start_date
        else:
            # also include historical weather data
            weather_start_date = collections.data_collections[weather_collection_name][
                "min_date"
            ]

    text = get_template_text()
    current_time = time.strftime("%y-%m-%d %H:%M:%S")
    text = text.replace("CURRENT_TIME", current_time)
    output_location_type = "azure" if run_mode == "batch" else "local"
    text = text.replace("COLLECTION_NAME", collection_name)
    text = text.replace("WEATHER_COLL_NAME", weather_collection_name)
    text = text.replace("OUTPUT_LOCATION_TYPE", output_location_type)
    text = text.replace("OUTPUT_LOCATION", output_location)
    text = text.replace("LATITUDE", latitude)
    text = text.replace("LONGITUDE", longitude)
    text = text.replace("PATTERN_TYPE", pattern_type)
    text = text.replace("START_DATE", start_date)
    text = text.replace("WEATHER_STARTDATE", weather_start_date)
    text = text.replace("END_DATE", end_date)
    text = text.replace("TIME_PER_POINT", time_per_point)
    text = text.replace("REGION_SIZE", region_size)
    text = text.replace("RUN_MODE", run_mode)
    text = text.replace("NUM_THREADS", str(n_threads))
    n_subimages = "10" if test_mode else "-1"
    text = text.replace("NUM_SUBIMAGES", n_subimages)
    if coords_id:
        text = text.replace("COORDS_ID_STRING", 'coords_id = "{}"'.format(coords_id))
    else:
        text = text.replace("COORDS_ID_STRING", "")
    with open(filename, "w") as configfile:
        configfile.write(text)
    print(
        "================================\nWrote file \n  {}\nWe recommend that you add and commit this to your version control repository.\n================================".format(
            filename
        )
    )
    return filename


def main():
    # get lists of options for the user to choose from.
    collection_names = [
        k
        for k in collections.data_collections.keys()
        if collections.data_collections[k]["data_type"] == "vegetation"
    ]
    run_modes = ["local", "batch"]
    date_regex = re.compile("[\d]{4}-[01][\d]-[0123][\d]")
    time_per_point_regex = re.compile("[\d]+[dwmy]")
    lat_range = [-90.0, 90.0]
    long_range = [-180.0, 180.0]
    n_threads_range = range(1, 17)
    default_n_threads = 4

    # create argument parser in case user wants to use command line args
    parser = argparse.ArgumentParser(
        description="""
        Create a config file for running pyveg_pipeline.  If run with no arguments (recommended), the user will be prompted for each parameter, or can choose a default value.
        """
    )
    parser.add_argument(
        "--coords_id", help="(optional) ID of location in coordinates.py", type=str
    )
    parser.add_argument(
        "--configs_dir", help="path to directory containing config files"
    )
    parser.add_argument("--collection_name", help="collection name (e.g. 'Sentinel2')")
    parser.add_argument(
        "--output_dir", help="Directory for local output data", type=str
    )
    parser.add_argument(
        "--test_mode",
        help="Run in test mode, over fewer months and with fewer sub-images",
        action="store_true",
    )
    parser.add_argument("--latitude", help="latitude in degrees N", type=float)
    parser.add_argument("--longitude", help="longitude in degrees E", type=float)
    parser.add_argument("--country", help="Country of location", type=str)
    parser.add_argument("--start_date", help="start date, format YYYY-MM-DD", type=str)
    parser.add_argument("--end_date", help="end date, format YYYY-MM-DD", type=str)
    parser.add_argument(
        "--time_per_point", help="frequency of image, e.g. '1m', '1w'", type=str
    )
    parser.add_argument(
        "--region_size",
        help="Size of region to download, in degrees lat/long",
        type=float,
    )
    parser.add_argument(
        "--pattern_type",
        help="Type of patterned vegetation, e.g. 'spots', 'labyrinths'",
        type=str,
    )
    parser.add_argument(
        "--run_mode",
        help="""
        'local' for running on local machine, 'azure' for running some time-consuming parts (i.e. vegetation image processing) on Azure batch
        """,
        type=str,
    )
    parser.add_argument(
        "--n_threads",
        help="""
        How many threads (cores) to parallelize some processing functions over
        """,
        type=int,
    )

    args = parser.parse_args()

    # sanity check
    if args.coords_id and (args.latitude or args.longitude):
        print("Please select EITHER coords_id OR latitude/longitude")
        return

    #############
    # now go through any arguments not already set via command line,
    # and prompt user for them.

    # configs_dir
    configs_dir = args.configs_dir if args.configs_dir else ""
    while not os.path.exists(configs_dir):
        if os.path.exists(os.path.join("pyveg", "configs")):
            default_configs_dir = os.path.join("pyveg", "configs")
        elif os.path.exists("configs"):
            default_configs_dir = "configs"
        else:
            default_configs_dir = "."
        configs_dir = input(
            "Enter path to directory containing config files, or press Return for default path ('{}') : ".format(
                default_configs_dir
            )
        )
        if len(configs_dir) == 0:
            configs_dir = default_configs_dir

    # test mode
    test_mode = args.test_mode if args.test_mode else False
    if not test_mode:
        do_test = input(
            "Would you like to make a test config file, with fewer months, and only a subset of sub-images?  Press 'y' if so, or press Return for a normal config. : "
        )
        test_mode = do_test.startswith("y") or do_test.startswith("Y")

    # collection name
    collection_name = args.collection_name if args.collection_name else None
    while not collection_name in collection_names:
        collection_name = input(
            "Please enter a valid collection name from this list: {} : ".format(
                collection_names
            )
        )

    # (optional) ID from coordinates.py
    coords_id = args.coords_id if args.coords_id else None
    latitude = None
    longitude = None
    country = None
    region_size = None
    pattern_type = None
    if coords_id:
        try:
            row = coordinate_store.loc[coords_id]
            latitude = row["latitude"]
            longitude = row["longitude"]
            country = row["country"]
            region_size = row["region_size"]
            pattern_type = row["type"]
        except (KeyError):
            print("Unknown id {} - please enter coordinates manually".format(coords_id))

    # latitude and longitude
    if not latitude:
        latitude = args.latitude if args.latitude else -999.0

        while not (
            isinstance(latitude, float)
            and latitude > lat_range[0]
            and latitude < lat_range[1]
        ):
            latitude = float(
                input(
                    "please enter Latitude (degrees N) in the range {} : ".format(
                        lat_range
                    )
                )
            )
    if not longitude:
        longitude = args.longitude if args.longitude else -999.0
        while not (
            isinstance(longitude, float)
            and longitude > long_range[0]
            and longitude < long_range[1]
        ):
            longitude = float(
                input(
                    "please enter Longitude (degrees E) in the range {} : ".format(
                        long_range
                    )
                )
            )

    # country
    country = args.country if args.country else ""
    if not country:
        country = input(
            "Enter name of country, or press return to use OpenCage country lookup based on coordinates : "
        )
        if len(country) == 0:
            country = lookup_country(latitude, longitude)
    # remove spaces
    country = re.sub("[\s]+", "", country)

    # start date
    start_date = args.start_date if args.start_date else ""
    if test_mode:
        default_start_date = "2019-01-01"
    else:
        default_start_date = collections.data_collections[collection_name]["min_date"]
    while not date_regex.search(start_date):
        start_date = input(
            "Enter start date in format YYYY-MM-DD, or press Return for default ({}) : ".format(
                default_start_date
            )
        )
        if len(start_date) == 0:
            start_date = default_start_date

    # end date
    end_date = args.end_date if args.end_date else ""
    if test_mode:
        default_end_date = "2019-03-01"
    else:
        default_end_date = collections.data_collections[collection_name]["max_date"]
    while not date_regex.search(end_date):
        end_date = input(
            "Enter end date in format YYYY-MM-DD, or press Return for default ({}) : ".format(
                default_end_date
            )
        )
        if len(end_date) == 0:
            end_date = default_end_date

    # time per point
    time_per_point = args.time_per_point if args.time_per_point else ""
    default_time_per_point = collections.data_collections[collection_name][
        "time_per_point"
    ]
    while not time_per_point_regex.search(time_per_point):
        time_per_point = input(
            "Enter time per point in format e.g. '1m' for 1 month, '1w' for 1 week, or press Return for default ({}) : ".format(
                default_time_per_point
            )
        )
        if len(time_per_point) == 0:
            time_per_point = default_time_per_point

    # region size
    if not region_size:
        region_size = args.region_size if args.region_size else -1.0
        default_region_size = 0.08
        while not (
            isinstance(region_size, float) and region_size > 0.0 and region_size <= 0.08
        ):
            region_size = input(
                "Enter region size in degrees latitude/longitude, or press Return for max/default ({}) : ".format(
                    default_region_size
                )
            )
            if len(region_size) == 0:
                region_size = default_region_size
            else:
                region_size = float(region_size)
    # now we've established it fulfils the requirements, convert to a str
    region_size = str(region_size)

    # pattern_type
    if not pattern_type:
        pattern_type = args.pattern_type if args.pattern_type else ""
        default_pattern_type = "unknown"
        while len(pattern_type) < 1:
            pattern_type = input(
                "Enter type of patterned vegetation (e.g. 'spots', 'labyrinths', or press Return for default ('{}') : ".format(
                    default_pattern_type
                )
            )
            if len(pattern_type) == 0:
                pattern_type = default_pattern_type
    pattern_type = pattern_type.replace(" ", "-").lower()

    # run mode
    run_mode = args.run_mode if args.run_mode else ""
    default_run_mode = "local"
    while not run_mode in run_modes:
        run_mode = input(
            "Would you like time-consuming functions to be run on the cloud?  Choose from the following: {}, or press Return for default option '{}': ".format(
                run_modes, default_run_mode
            )
        )
        if len(run_mode) == 0:
            run_mode = default_run_mode

    # output directory
    output_dir = args.output_dir if args.output_dir else ""
    if run_mode == "local" and not output_dir:
        output_dir = input(
            "Enter location for output, or press Return for default ('.') : "
        )
        if len(output_dir) == 0:
            output_dir = "."

    lat_string = "{:.2f}".format(latitude)
    long_string = "{:.2f}".format(longitude)
    output_location = make_output_location(
        coords_id, collection_name, lat_string, long_string, country
    )

    if run_mode == "local":
        output_location = os.path.join(output_dir, output_location)

    # num threads
    n_threads = args.n_threads if args.n_threads else 0
    while not (isinstance(n_threads, int) and n_threads in n_threads_range):
        if run_mode == "local":
            n_threads = input(
                "How many threads would you like time-consuming processing functions to use?  (Many computers will have 4 or 8 threads available).  Press return for default value {} : ".format(
                    default_n_threads
                )
            )
            if len(n_threads) == 0:
                n_threads = default_n_threads
            else:
                try:
                    n_threads = int(n_threads)
                except:
                    print("Please enter an integer value")
        else:
            n_threads = 1

    print(
        """
    output_location {}
    collection: {}
    latitude: {}
    longitude: {}
    country: {}
    pattern_type: {}
    start_date: {}
    end_date: {}
    time_per_point: {}
    region_size: {}
    run_mode: {}
    n_threads: {}
    """.format(
            output_location,
            collection_name,
            lat_string,
            long_string,
            country,
            pattern_type,
            start_date,
            end_date,
            time_per_point,
            region_size,
            run_mode,
            n_threads,
        )
    )

    config_filename = write_file(
        configs_dir,
        output_location,
        long_string,
        lat_string,
        country,
        pattern_type,
        start_date,
        end_date,
        time_per_point,
        region_size,
        collection_name,
        run_mode,
        n_threads,
        test_mode,
        coords_id,
    )
    print(
        """
To run pyveg using this configuration, do:

pyveg_run_pipeline --config_file {}

""".format(
            config_filename
        )
    )


if __name__ == "__main__":
    main()
