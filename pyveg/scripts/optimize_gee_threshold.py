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
* Count how many sub-images have > 50% difference between them.
* Return the number of "good" sub-images (not all-black or all-white or
too much variation.


"""

import os

import argparse

from pyveg.src.satellite_data_analysis import get_time_series


def count_good_images(filebase):
    num_good = 0


def main():
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
    parser.add_argument("--num_time_points",help="Get a time series with this many divisions between start_date and end_date", type=int, default=1)
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

    args = parser.parse_args()
    sanity_check_args(args)

    image_coll = args.image_coll
    start_date = args.start_date
    end_date = args.end_date
    output_dir = args.output_dir
    output_suffix = args.output_suffix
    bands = args.bands.split(",")
    region_size = args.region_size
    scale = args.scale
    mask_cloud = True if args.mask_cloud else False
    input_file = arg.input_file if args.input_file else None
    num_time_points = args.num_time_points
    if args.coords_point:
        coords = [float(x) for x in args.coords_point.split(",")]
    elif args.coords_rect:
        coords_all = [float(x) for x in args.coords_rect.split(",")]
        coords = [ [coords_all[2*i],coords_all[2*i+1]] for i in range(int(len(coords_all)/2))]
    else:
        coords = None
    ##
    get_time_series(num_time_points,
                    input_file,
                    coords,
                    image_coll,
                    bands,
                    region_size,
                    scale,
                    start_date,
                    end_date,
                    mask_cloud,
                    output_dir,
                    output_suffix)
    print("Done")




if __name__ == "__main__":
    main()
