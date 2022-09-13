#!/usr/bin/env python

"""
Generate a config file peep/configs/<config_filename> for use when running
download and processing jobs with
peep_run_pipeline --config_file peep/configs/<config_filename>

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

peep_generate_config

then respond to prompts, or

peep_generate_config --help

to see a list of command line options.
(Note that command line options and prompted inputs can be mixed-and-matched).

"""

import argparse
import os
import re
import time

import geopandas

from peep.configs import collections


def get_template_text():
    template_filepath = os.path.join(
        os.path.dirname(__file__), "..", "configs", "config_template.py"
    )
    if not os.path.exists(template_filepath):
        raise RuntimeError("Unable to find template {}".format(template_filepath))
    return open(template_filepath).read()


def make_output_location(coords_id, collection_name, left, bottom, right, top):
    if coords_id:
        output_location = f"{coords_id}-{collection_name}-{left}-{bottom}-{right}-{top}"
    else:
        output_location = f"{collection_name}-{left}-{bottom}-{right}-{top}"
    return output_location


def make_filename(
    configs_dir,
    left,
    bottom,
    right,
    top,
    start_date,
    end_date,
    time_per_point,
    collection_name,
    coords_id,
):
    """
    Construct a filename from the specified parameters.
    """
    filename_start = "config"
    if coords_id:
        filename_start += "_" + str(coords_id)
    filepath = os.path.join(
        configs_dir,
        f"{filename_start}_{collection_name}_{left}_{bottom}_{right}_{top}_{start_date}_{end_date}_{time_per_point}.py",
    )
    return filepath


def write_file(
    configs_dir,
    output_location,
    left,
    bottom,
    right,
    top,
    start_date,
    end_date,
    time_per_point,
    collection_name,
    coords_id=None,
):
    """
    Take the arguments, construct a filename, and write contents
    """
    filename = make_filename(
        configs_dir,
        left,
        bottom,
        right,
        top,
        start_date,
        end_date,
        time_per_point,
        collection_name,
        coords_id,
    )

    text = get_template_text()
    current_time = time.strftime("%y-%m-%d %H:%M:%S")
    text = text.replace("CURRENT_TIME", current_time)
    text = text.replace("COLLECTION_NAME", collection_name)
    text = text.replace("OUTPUT_LOCATION", output_location)
    text = text.replace(
        "RIGHT", str(int(right))
    )  # hacky way of removing unnecesary zeros
    text = text.replace("LEFT", str(int(left)))
    text = text.replace("TOP", str(int(top)))
    text = text.replace("BOTTOM", str(int(bottom)))
    text = text.replace("START_DATE", start_date)
    text = text.replace("END_DATE", end_date)
    text = text.replace("TIME_PER_POINT", time_per_point)

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
    date_regex = re.compile("[\d]{4}-[01][\d]-[0123][\d]")
    time_per_point_regex = re.compile("[\d]+[dwmy]")
    left_bound = [0, 700000]
    bottom_bound = [0, 1300000]
    top_bound = [0, 1300000]
    right_bound = [0, 700000]

    # create argument parser in case user wants to use command line args
    parser = argparse.ArgumentParser(
        description="""
        Create a config file for running peep_pipeline.  If run with no arguments (recommended), the user will be prompted for each parameter, or can choose a default value.
        """
    )
    parser.add_argument(
        "--bounds_file",
        help="Path to a geoparket file. The file should include a geometry column (named 'geometry'). A config file will be created for each row in the geoparket file.",
        type=str,
    )
    parser.add_argument(
        "--configs_dir", help="path to directory containing config files"
    )
    parser.add_argument(
        "--collection_name",
        help="collection name (e.g. 'Sentinel2')",
        default="Sentinel2",
    )
    parser.add_argument(
        "--output_dir", help="Directory for local output data", type=str
    )
    parser.add_argument("--left", help="left bound in Eastings", type=int)
    parser.add_argument("--right", help="right bound in Eastings", type=int)
    parser.add_argument("--bottom", help="bottom bound in Nothings", type=int)
    parser.add_argument("--top", help="top bound in Nothings", type=int)
    parser.add_argument("--start_date", help="start date, format YYYY-MM-DD", type=str)
    parser.add_argument("--end_date", help="end date, format YYYY-MM-DD", type=str)
    parser.add_argument(
        "--time_per_point", help="frequency of image, e.g. '1m', '1w'", type=str
    )

    args = parser.parse_args()

    # sanity check
    if args.bounds_file and (args.bottom or args.right or args.left or args.right):
        print("Please select EITHER coords_id OR bounds")
        return

    #############
    # now go through any arguments not already set via command line,
    # and prompt user for them.

    # configs_dir
    configs_dir = args.configs_dir if args.configs_dir else ""

    if configs_dir == "":
        if os.path.exists(os.path.join("peep", "configs")):
            default_configs_dir = os.path.join("peep", "configs")
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

    else:
        if not os.path.exists(configs_dir):
            os.makedirs(configs_dir, exist_ok=False)

    # collection name
    collection_name = args.collection_name if args.collection_name else None
    while not collection_name in collection_names:
        collection_name = input(
            "Please enter a valid collection name from this list: {} : ".format(
                collection_names
            )
        )
    # start date
    start_date = args.start_date if args.start_date else ""
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
        # output directory
    output_dir = args.output_dir if args.output_dir else ""
    if not output_dir:
        output_dir = input(
            "Enter location for output, or press Return for default ('.') : "
        )
        if len(output_dir) == 0:
            output_dir = "."

    # (optional) ID from coordinates.py
    bounds_file = args.bounds_file if args.bounds_file else None
    left = None
    right = None
    bottom = None
    top = None
    if bounds_file:
        bounds_gdf = geopandas.read_parquet(bounds_file)
        bounds_gdf.to_crs("EPSG:27700")

        if "on_land" in bounds_gdf:
            index = bounds_gdf[bounds_gdf["on_land"] == True].index
        else:
            index = bounds_gdf.index

        for i in index:
            row = bounds_gdf.iloc[i]
            bottom = int(row["geometry"].bounds[1])
            left = int(row["geometry"].bounds[0])
            right = int(row["geometry"].bounds[2])
            top = int(row["geometry"].bounds[3])

            left_string = "{:0>6}".format(left)
            bottom_string = "{:0>7}".format(bottom)
            right_string = "{:0>6}".format(right)
            top_string = "{:0>7}".format(top)

            output_location = make_output_location(
                i, collection_name, left_string, bottom_string, right_string, top_string
            )

            output_location = os.path.join(output_dir, output_location)

            print(
                """
                    output_location {}
                    collection: {}
                    left: {}
                    bottom: {}
                    right: {}
                    top: {}
                    start_date: {}
                    end_date: {}
                    time_per_point: {}
                      """.format(
                    output_location,
                    collection_name,
                    left_string,
                    bottom_string,
                    right_string,
                    top_string,
                    start_date,
                    end_date,
                    time_per_point,
                )
            )

            config_filename = write_file(
                configs_dir,
                output_location,
                left_string,
                bottom_string,
                right_string,
                top_string,
                start_date,
                end_date,
                time_per_point,
                collection_name,
                i,
            )

            print(
                """
                    To run peep using this configuration, do:

                    peep_run_pipeline --config_file {}
                    """.format(
                    config_filename
                )
            )
    else:
        # bounds
        if not left:
            left = args.left if args.left else 0

            while not (
                isinstance(left, int)
                and left >= left_bound[0]
                and left <= left_bound[1]
            ):
                left = float(
                    input(
                        "please enter left bound (eastings) in the range {} : ".format(
                            left_bound[1]
                        )
                    )
                )
        if not right:
            right = args.right if args.right else 0
            while not (
                isinstance(right, int)
                and right >= right_bound[0]
                and right <= right_bound[1]
            ):
                right = float(
                    input(
                        "please enter right bound (eastings) in the range {} : ".format(
                            right[1]
                        )
                    )
                )
        if not top:
            top = args.top if args.top else 0
            while not (
                isinstance(top, int) and top >= top_bound[0] and top <= top_bound[1]
            ):
                top = float(
                    input(
                        "please enter top bound (degrees northings) in the range {} : ".format(
                            top_bound[1]
                        )
                    )
                )
        if not bottom:
            bottom = args.bottom if args.bottom else 0
            while not (
                isinstance(bottom, int)
                and bottom >= bottom_bound[0]
                and top <= bottom_bound[1]
            ):
                bottom = float(
                    input(
                        "please enter bottom bound (degrees northings) in the range {} : ".format(
                            bottom_bound[1]
                        )
                    )
                )

        left_string = "{:0>6}".format(left)
        bottom_string = "{:0>7}".format(bottom)
        right_string = "{:0>6}".format(right)
        top_string = "{:0>7}".format(top)

        output_location = make_output_location(
            None, collection_name, left_string, bottom_string, right_string, top_string
        )

        print(
            """
        output_location {}
        collection: {}
        left: {}
        bottom: {}
        right: {}
        top: {}
        start_date: {}
        end_date: {}
        time_per_point: {}
          """.format(
                output_location,
                collection_name,
                left_string,
                bottom_string,
                right_string,
                top_string,
                start_date,
                end_date,
                time_per_point,
            )
        )

        config_filename = write_file(
            configs_dir,
            output_location,
            left_string,
            bottom_string,
            right_string,
            top_string,
            start_date,
            end_date,
            time_per_point,
            collection_name,
        )
        print(
            """
        To run peep using this configuration, do:

        peep_run_pipeline --config_file {}

        """.format(
                config_filename
            )
        )


if __name__ == "__main__":
    main()
