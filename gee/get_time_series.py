#!/usr/bin/env python

"""
Generate a time_series of images for a given location.
"""

import os
import argparse
import dateparser
from datetime import datetime, timedelta


from download_images import *
from image_utils import *


def divide_time_period(start_date, end_date, n):
    """
    Divide the full period between the start_date and end_date into n equal-length
    (to the nearest day) chunks.
    Takes start_date and end_date as strings 'YYYY-MM-DD'.
    Returns a list of tuples
    [ (chunk0_start,chunk0_end),...]
    """
    start = dateparser.parse(start_date)
    end = dateparser.parse(end_date)
    if (not isinstance(start, datetime)) or (not isinstance(end, datetime)):
        raise RuntimeError("invalid time strings")
    td = end - start
    if td.days <= 0:
        raise RuntimeError("end_date must be after start_date")
    days_per_chunk = td.days // n
    output_list = []
    for i in range(n):
        chunk_start = start + timedelta(days=(i*days_per_chunk))
        chunk_end = start + timedelta(days=((i+1)*days_per_chunk))
        ## unless we are in the last chunk, which should finish at end_date
        if i == n-1:
            chunk_end = end
        output_list.append((chunk_start.isoformat().split("T")[0],
                           chunk_end.isoformat().split("T")[0]))
    return output_list




if __name__ == "__main__":
    """
    use command line arguments to choose images.
    """
    parser = argparse.ArgumentParser(description="download from EE")
    parser.add_argument("--image_coll",help="image collection",
                        default="LANDSAT/LC08/C01/T1_SR")
    parser.add_argument("--start_date",help="YYYY-MM-DD",
                        default="2013-03-30")
    parser.add_argument("--end_date",help="YYYY-MM-DD",
                        default="2013-04-01")
    parser.add_argument("--coords_point",help="'long,lat'")
    parser.add_argument("--coords_rect",help="'long1,lat1,long2,lat2...,...'")
    parser.add_argument("--bands",help="string containing comma-separated list",
                        default="B2,B3,B4,B5,B6,B7")
    parser.add_argument("--region_size", help="size of output region in long/lat", default=0.1, type=float)
    parser.add_argument("--scale", help="size of each pixel in output image (m)", default=10, type=int)
    parser.add_argument("--output_dir",help="output directory",
                        default=".")
    parser.add_argument("--output_suffix",help="end of output filename, including file extension",
                      default="gee_img.png")
    parser.add_argument("--input_file",help="text file with coordinates, one per line")
    parser.add_argument("--mask_cloud",help="EXPERIMENTAL - apply cloud masking function",action='store_true')
    args = parser.parse_args()
    sanity_check_args(args)
