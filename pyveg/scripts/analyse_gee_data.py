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

from pyveg.src.satellite_data_analysis import get_time_series,divide_time_period_in_n_day_portions



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
    parser.add_argument("--num_days_per_point",help="Get a time series with using portions of this number of days between start_date and end_date. This option supersedes the -num_time_points option", type=int,default=0)

    parser.add_argument("--coords",help="'long,lat'")
    parser.add_argument("--bands",help="string containing comma-separated list",
                        default="B2,B3,B4,B5,B6,B7")
    parser.add_argument("--region_size", help="size of output region in long/lat", default=0.1, type=float)
    parser.add_argument("--scale", help="size of each pixel in output image (m)", default=10, type=int)
    parser.add_argument("--output_dir",help="output directory",
                        default=".")
    parser.add_argument("--output_suffix",help="end of output filename, including file extension",
                      default="gee_img.png")
    parser.add_argument("--mask_cloud",help="EXPERIMENTAL - apply cloud masking function",action='store_true')
    parser.add_argument("--network_centrality",help="calculate network centrality measures on images and print them out as json files",action='store_true')

    args = parser.parse_args()

    image_coll = args.image_coll
    start_date = args.start_date
    end_date = args.end_date
    output_dir = args.output_dir
    output_suffix = args.output_suffix
    bands = args.bands.split(",")
    region_size = args.region_size
    scale = args.scale
    mask_cloud = True if args.mask_cloud else False
    network_centrality = True if args.network_centrality else False

    num_time_points = args.num_time_points

    # if the --num_days_per_point option exists, overwrite any existing option from num_time_points
    if args.num_days_per_point!=0:
        num_time_points = divide_time_period_in_n_day_portions(start_date,end_date,args.num_days_per_point)

    coords = [float(x) for x in args.coords.split(",")]

    ##
    get_time_series(num_time_points,
                    coords,
                    image_coll,
                    bands,
                    region_size,
                    scale,
                    start_date,
                    end_date,
                    mask_cloud,
                    output_dir,
                    output_suffix,
                    network_centrality)


    print("Done")




if __name__ == "__main__":
    main()
