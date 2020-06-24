#!/usr/bin/env python

"""
Script to download images from Google Earth Engine.

You need to have the earthengine-api installed
```
 pip install earthengine-api
```
and before running the script, from the command-line, do:
```
 earthengine authenticate
```
and follow the instructions there.

The procedure is then to either loop through an input file containing
a list of coordinates (one per line in format:   lat,long ) OR receive an
individual set of coordinates as a command line argument, then:
* Find image
* Filter clouds
* Get download URL
* Download zipfile
* Unpack zipfile
* Combine tif files for individual bands into one output file

Needs a relatively recent version of pillow (fork of PIL):
```
pip install --upgrade pillow
```
"""

import os
import argparse
import warnings
import time
from shutil import copyfile

from pyveg.src.process_satellite_data import process_all_collections
from pyveg import config
from pyveg.coordinates import coordinate_store


def main():
    """
    use command line arguments to choose images.
    """

    help_text = """Options should be specified in the config 
    file where possible, but can be overwritten using this CLI.
    
    Example usage:

    python download_gee_data.py \\
        --start_date 2016-01-01 \\
        --end_date 2019-01-01 \\
        --coords 27.99,11.29 \\
    """

    # crate argparse
    parser = argparse.ArgumentParser(
        description="download from EE",
        epilog=help_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # add arguments - keep it minimal here, but overwrite value from config
    # if option is given here
    parser.add_argument("--start_date", help="YYYY-MM-DD")
    parser.add_argument("--end_date", help="YYYY-MM-DD")
    parser.add_argument("--coordinates", help="'long,lat'")

    args = parser.parse_args()

    # overwrite dates if specified
    if args.start_date is not None and args.end_date is not None:
        config.date_range = (args.start_date, args.end_date)

    # overwrite coords if specified
    if args.coordinates is not None:
        coordinates = args.coordinates.split(",")
        config.coordinates = (float(coordinates[0]), float(coordinates[1]))

    # parse output directory
    config.output_dir += "__" + time.strftime("%Y-%m-%d_%H-%M-%S")
    config.output_dir = os.path.join(
        "output", config.output_dir
    )  # force into .gitignore

    # before we run anythin, save the current config to the output dir
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)
    copyfile(config.__file__, os.path.join(config.output_dir, "config_cached.py"))

    # print which collections we are running with
    print("-" * 35)
    print("Running download_gee_data.py")
    print("-" * 35)

    # print job configuration to the user
    print("Job Configuration:")

    # search coordinate store to see if we have an entry for these coordinates
    long, lat = config.coordinates
    entry = coordinate_store[
        (coordinate_store["longitude"] == long) & (coordinate_store["latitude"] == lat)
    ]

    if entry.empty:
        print(f"* Warning: running with unknown coordinates: {config.coordinates}.")
    elif len(entry) == 1:
        print(f"* Running with location id {entry.index[0]}:")
        print(entry.to_string(index=False))
    else:
        raise RuntimeError("Coordinate store error.")

    print(f"* date range:\t {config.date_range}")
    collection_names = [c["collection_name"] for c in config.data_collections.values()]
    print(f"* collections:\t {collection_names}\n")

    # run!
    process_all_collections(
        config.output_dir,
        config.data_collections,
        config.coordinates,
        config.date_range,
    )

    print("\nFinished all. To analyse results, run:")
    print(f"> pyveg_gee_analysis --input_dir {config.output_dir}\n")


if __name__ == "__main__":
    main()
