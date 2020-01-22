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

The idea is to download two images for a given set of coords,
and a short time between them, then run the convert-to-BW,
divide into sub-images, calculate Euler characteristic procedure
on them, then:
* Count how many sub-images are all-black or all-white.
* Count how many sub-images have > 20% difference between them.
* Return the number of "good" sub-images (not all-black or all-white or
too much variation.
"""

import os
import shutil
import re
import argparse
from PIL import Image

from pyveg.src.satellite_data_analysis import get_time_series
from pyveg.src.image_utils import image_file_all_same_colour, compare_binary_image_files

if os.name == "posix":
    TMPDIR = "/tmp/"
else:
    TMPDIR = "%TMP%"


def count_good_images(date_loc_dict):
    """
    Go through the date_loc dict and count how many
    images are "bad"
    """
    num_total = 0
    num_all_black = 0
    num_all_white = 0
    num_big_change = 0

    for k,v in date_loc_dict.items():
        both_single_colour = True
        for date, img in v.items():
            num_total += 1
            is_all_black = image_file_all_same_colour(img, (0,0,0))
            is_all_white = image_file_all_same_colour(img, (255,255,255))
            if is_all_black:
                num_all_black += 1
            elif is_all_white:
                num_all_white += 1
            both_single_colour &= (is_all_white | is_all_black)
        dates = list(v.keys())
        if len(dates) < 2:
            print("Need at least two dates to compare")
            continue
        if not both_single_colour:
            is_similar = compare_binary_image_files(v[dates[0]],v[dates[1]])
            if is_similar < 0.9:
                num_big_change += 2

    return 1.0 - float((num_all_black + num_all_white + num_big_change) / num_total)


def create_date_location_dict(input_dir):
    """
    Loop through directory, gather filenames that have
    the same coordinates and different dates.
    """
    # list the files in the directory.  see how many dates there are:
    date_regex = re.compile("([\d]{4}-[\d]{2}-[\d]{2})")
    coord_regex = re.compile("([\d]{1,3}\.[\d]{1,3}_[\d]{1,3}\.[\d]{1,3})")
    dates_locations = {}
    for f in os.listdir(input_dir):
        if not f.endswith("png"):
            continue
        loc_match = coord_regex.search(f)
        if not loc_match:
            print("Couldnt find coordinates in {}".format(f))
            continue
        coords = loc_match.groups()[0]
        if not coords in dates_locations.keys():
            dates_locations[coords] = {}
        date_match = date_regex.search(f)
        if not date_match:
            continue
        date = date_match.groups()[0]
        if not date in dates_locations[coords].keys():
            dates_locations[coords][date] = os.path.join(input_dir,f)
    return dates_locations


def optimize_threshold(threshold,
             start_date="2016-03-30",
             end_date="2016-04-30",
             coords=[27.95,11.57]):
    """
    Run the get_time_series method, making some assumptions about
    what image_coll, bands, etc we will use.
    """
    image_coll = "COPERNICUS/S2"
    num_time_points = 2
    bands = ["NDVI"]
    region_size = 0.1
    scale = 10
    opt_dir = os.path.join(TMPDIR,"optimize_{}".format(threshold))
    if os.path.exists(opt_dir):
        shutil.rmtree(opt_dir)
    mask_cloud = False
    get_time_series(num_time_points,
                    coords,
                    image_coll,
                    bands,
                    region_size,
                    scale,
                    start_date,
                    end_date,
                    mask_cloud,
                    opt_dir,
                    network_centrality=False,
                    threshold=threshold)
    # this will havce put a bunch of files in opt_dir
    # now sort them by coordinates and date
    date_location_dir = create_date_location_dict(opt_dir)
    # then look at them to count the good ones
    numbers = count_good_images(date_location_dir)
    return numbers


def main():
    """
    use command line arguments to choose images.
    """
    parser = argparse.ArgumentParser(description="download from EE")
    parser.add_argument("--start_date",help="YYYY-MM-DD",
                        default="2016-03-30")
    parser.add_argument("--end_date",help="YYYY-MM-DD",
                        default="2016-04-30")
    parser.add_argument("--coords",help="'long,lat'", default="27.95,11.57")
    parser.add_argument("--threshold",help="threshold (0-765)", default=470)

    args = parser.parse_args()
    #sanity_check_args(args)

    start_date = args.start_date
    end_date = args.end_date
    coords = [float(x) for x in args.coords.split(",")]
    threshold = args.threshold
    optimize_threshold(threshold, start_date, end_date, coords)
    print("Done")


if __name__ == "__main__":
    main()
