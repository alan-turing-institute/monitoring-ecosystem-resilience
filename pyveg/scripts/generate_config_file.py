#!/usr/bin/env python

"""
Generate a config file pyveg/configs/<config_filename> for use when running
download and processing jobs with
pyveg_run_pipeline --config_file pyveg/configs/<config_filename>

User specifies:
* Coordinates
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

import os
import re
import argparse
import time

from pyveg.configs import collections
from pyveg.src.coordinate_utils import lookup_country

def get_template_text():
    template_filepath = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "config_template.py"
    )
    if not os.path.exists(template_filepath):
        raise RuntimeError("Unable to find template {}".format(template_filepath))
    return open(template_filepath).read()


def make_output_location(collection_name,
                         latitude,
                         longitude,
                         country):
    return f"{collection_name}-{latitude}N-{longitude}E-{country}"

def make_filename(configs_dir,
                  test_mode,
                  longitude,
                  latitude,
                  country,
                  start_date,
                  end_date,
                  time_per_point,
                  collection_name,
                  run_mode):
    """
    Construct a filename from the specified parameters.
    """
    filename_start = "testconfig" if test_mode else "config"
    filepath = os.path.join(
        configs_dir,
        f"{filename_start}_{collection_name}_{latitude}N_{longitude}E_{country}_{start_date}_{end_date}_{time_per_point}_{run_mode}.py"
        )
    return filepath


def write_file(configs_dir,
               output_location,
               longitude,
               latitude,
               country,
               start_date,
               end_date,
               time_per_point,
               collection_name,
               run_mode,
               n_threads,
               test_mode=False):
    """
    Take the arguments, construct a filename, and write contents
    """
    filename = make_filename(configs_dir,
                             test_mode,
                             longitude,
                             latitude,
                             country,
                             start_date,
                             end_date,
                             time_per_point,
                             collection_name,
                             run_mode)

    print("Will write file \n {} \n".format(filename))
    text = get_template_text()
    current_time = time.strftime("%y-%m-%d %H:%M:%S")
    text = re.sub("CURRENT_TIME", current_time, text)
    output_location_type = "azure" if run_mode == "batch" else "local"
    text = re.sub("COLLECTION_NAME", collection_name, text)
    text = re.sub("OUTPUT_LOCATION_TYPE", output_location_type, text)
    text = re.sub("OUTPUT_LOCATION", output_location, text)
    text = re.sub("LATITUDE", latitude, text)
    text = re.sub("LONGITUDE", longitude, text)
    text = re.sub("START_DATE", start_date, text)
    text = re.sub("END_DATE", end_date, text)
    text = re.sub("TIME_PER_POINT", time_per_point, text)
    text = re.sub("RUN_MODE", run_mode, text)
    text = re.sub("NUM_THREADS", str(n_threads), text)
    n_subimages = '10' if test_mode else '-1'
    text = re.sub("NUM_SUBIMAGES", n_subimages, text)
    with open(filename, "w") as configfile:
        configfile.write(text)
    print("Wrote file \n{}\n  We recommend that you add and commit this to your version control repository.".format(filename))


def main():
    collection_names = collections.data_collections.keys()
    #["Sentinel2","Landsat4","Landsat7","Landsat8"]
    run_modes = ["local","batch"]
    date_regex = re.compile("[\d]{4}-[01][\d]-[0123][\d]")
    time_per_point_regex = re.compile("[\d]+[dwmy]")
    lat_range = [-90.,90.]
    long_range = [-180., 180.]
    n_threads_range = range(1,17)
    default_n_threads = 4
    parser = argparse.ArgumentParser(
        description="create a config file for running pyveg_pipeline"
    )
    parser.add_argument(
        "--configs_dir", help="path to directory containing config files"
    )
    parser.add_argument(
        "--collection_name", help="collection name (e.g. 'Sentinel2')"
    )
    parser.add_argument(
        "--output_dir", help="Directory for local output data", type=str
    )
    parser.add_argument(
        "--test_mode", help="Run in test mode, over fewer months and with fewer sub-images", action='store_true'
    )
    parser.add_argument(
        "--latitude", help="latitude in degrees N", type=float
    )
    parser.add_argument(
        "--longitude", help="longitude in degrees E", type=float
    )
    parser.add_argument(
        "--country", help="Country of location", type=str
    )
    parser.add_argument(
        "--start_date", help="start date, format YYYY-MM-DD", type=str
    )
    parser.add_argument(
        "--end_date", help="end date, format YYYY-MM-DD", type=str
    )
    parser.add_argument(
        "--time_per_point", help="frequency of image, e.g. '1m', '1w'", type=str
    )
    parser.add_argument(
        "--run_mode", help="""
        'local' for running on local machine, 'azure' for running some time-consuming parts (i.e. vegetation image processing) on Azure batch
        """, type=str
    )
    parser.add_argument(
        "--n_threads", help="""
        How many threads (cores) to parallelize some processing functions over
        """, type=int
    )

    args = parser.parse_args()

    # configs_dir
    configs_dir = args.configs_dir if args.configs_dir else ""
    while not os.path.exists(configs_dir):
        if os.path.exists(os.path.join("pyveg","configs")):
            default_configs_dir = os.path.join("pyveg","configs")
        elif os.path.exists("configs"):
            default_configs_dir = "configs"
        else:
            default_configs_dir = "."
        configs_dir = input("Enter path to directory containing config files, or press Return for default path '{}' : ".format(default_configs_dir))
        if len(configs_dir) == 0:
            configs_dir = default_configs_dir

    # test mode
    test_mode = args.test_mode if args.test_mode else False
    if not test_mode:
        do_test = input("Would you like to make a test config file, with fewer months, and only a subset of sub-images?  Press 'y' if so, or press Return for a normal config. : ")
        test_mode = do_test.startswith("y") or do_test.startswith("Y")

    # collection name
    collection_name = args.collection_name if args.collection_name else None
    while not collection_name in collection_names:
        collection_name = input("Please enter a valid collection name from this list: {} : ".format(collection_names))

    # latitude and longitude
    latitude = args.latitude if args.latitude else -999.
    print("latitude {} lat_range[0] {} lat_range[1]".format(latitude, lat_range[0], lat_range[1]))
    while not (isinstance(latitude, float) and latitude > lat_range[0] and latitude < lat_range[1]):
        latitude = float(input("please enter Latitude (degrees N) in the range {} : ".format(lat_range)))
    longitude = args.longitude if args.longitude else -999.
    while not (isinstance(longitude, float) and longitude > long_range[0] and longitude < long_range[1]):
        longitude = float(input("please enter Longitude (degrees E) in the range {} : ".format(long_range)))

    # country
    country = input("Enter name of country, or press return to use OpenCage country lookup based on coordinates : ")
    if len(country) == 0:
        country = lookup_country(latitude, longitude)
    # remove spaces
    country = re.sub("[\s]+","",country)

    # start date
    start_date = args.start_date if args.start_date else ""
    if test_mode:
        default_start_date = "2019-01-01"
    else:
        default_start_date = collections.data_collections[collection_name]["min_date"]
    while not date_regex.search(start_date):
        start_date = input("Enter start date in format YYYY-MM-DD, or press Return for default ({}) : ".format(default_start_date))
        if len(start_date) == 0:
            start_date = default_start_date

    # end date
    end_date = args.end_date if args.end_date else ""
    if test_mode:
        default_end_date = "2019-03-01"
    else:
        default_end_date = collections.data_collections[collection_name]["max_date"]
    while not date_regex.search(end_date):
        end_date = input("Enter end date in format YYYY-MM-DD, or press Return for default ({}) : ".format(default_end_date))
        if len(end_date) == 0:
            end_date = default_end_date

    # time per point
    time_per_point = args.time_per_point if args.time_per_point else ""
    default_time_per_point = collections.data_collections[collection_name]["time_per_point"]
    while not time_per_point_regex.search(time_per_point):
        time_per_point = input("Enter time per point in format e.g. '1m' for 1 month, '1w' for 1 week, or press Return for default ({}) : ".format(default_time_per_point))
        if len(time_per_point) == 0:
            time_per_point = default_time_per_point

    # run mode
    run_mode = args.run_mode if args.run_mode else ""
    default_run_mode = "local"
    while not run_mode in run_modes:
        run_mode = input("Would you like time-consuming functions to be run on the cloud?  Choose from the following: {}, or press Return for default option '{}': ".format(run_modes, default_run_mode))
        if len(run_mode) == 0:
            run_mode = default_run_mode

    # output directory
    output_dir = args.output_dir if args.output_dir else ""
    if run_mode == "local" and not output_dir:
        output_dir = input("Enter location for output, or press Return for default ('.') : ")
        if len(output_dir) == 0:
                output_dir = "."

    lat_string = "{:.2f}".format(latitude)
    long_string = "{:.2f}".format(longitude)
    output_location = make_output_location(collection_name,
                                           lat_string,
                                           long_string,
                                           country)

    if run_mode == "local":
        output_location = os.path.join(output_dir, output_location)


    # num threads
    n_threads = args.n_threads if args.n_threads else 0
    while not (isinstance(n_threads, int) and n_threads in n_threads_range):
        if run_mode == "local":
            n_threads = input("How many threads would you like time-consuming processing functions to use?  (Many computers will have 4 or 8 threads available).  Press return for default value {} : ".format(default_n_threads))
            if len(n_threads) == 0:
                n_threads = default_n_threads
            else:
                try:
                    n_threads = int(n_threads)
                except:
                    print("Please enter an integer value")
        else:
            n_threads = 1

    print("""
    output_location {}
    collection: {}
    latitude: {}
    longitude: {}
    country: {}
    start_date: {}
    end_date: {}
    time_per_point: {}
    run_mode: {}
    n_threads: {}
    """.format(output_location, collection_name, lat_string, long_string, country, start_date, end_date, time_per_point, run_mode, n_threads))

    write_file(configs_dir,
               output_location,
               long_string,
               lat_string,
               country,
               start_date,
               end_date,
               time_per_point,
               collection_name,
               run_mode,
               n_threads,
               test_mode)


if __name__ == "__main__":
    main()
