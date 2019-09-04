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
"""


import os
import sys
import requests
import argparse
from zipfile import ZipFile

from PIL import Image

import ee
ee.Initialize()

def scale_tif(input_filebase, bands=["B4","B3","B2"]):
    """
    Read tif files in "I" mode - one per specified band, and rescale and combine
    pixel values to r,g,b values betweek 0 and 255 in a combined output image.
    """

    band_dict = {"r": {"band": bands[0],
                       "min_val": sys.maxsize,
                       "max_val": -1*sys.maxsize,
                       "pix_vals": []},
                 "g": {"band": bands[1],
                       "min_val": sys.maxsize,
                       "max_val": -1*sys.maxsize,
                       "pix_vals": []},
                 "b": {"band": bands[2],
                       "min_val": sys.maxsize,
                       "max_val": -1*sys.maxsize,
                       "pix_vals": []}
                }

    for col in band_dict.keys():
        im = Image.open(input_filebase+"."+band_dict[col]["band"]+".tif")
        pix = im.load()
        ## find the minimum and maximum pixel values in the original scale
        for ix in range(im.size[0]):
            for iy in range(im.size[1]):
                if pix[ix,iy]> band_dict[col]["max_val"]:
                    band_dict[col]["max_val"]= pix[ix,iy]
                elif pix[ix,iy] < band_dict[col]["min_val"]:
                    band_dict[col]["min_val"] = pix[ix,iy]
        band_dict[col]["pix_vals"] = pix
    # Take the overall max of the three bands to be the value to scale down with.
    overall_max = max((band_dict[col]["max_val"] for col in ["r","g","b"]))
    print("{},{} {},{} {},{}".format(band_dict["r"]["max_val"],band_dict["r"]["min_val"],
                                     band_dict["g"]["max_val"],band_dict["g"]["min_val"],
                                     band_dict["b"]["max_val"],band_dict["b"]["min_val"]))
    # create a new image where we will fill RGB pixel values from 0 to 255
    get_pix_val = lambda ix, iy, col: \
        max(0, int(band_dict[col]["pix_vals"][ix,iy] * 255/ \
#                   band_dict[col]["max_val"]
                   overall_max
        ))
    new_img = Image.new("RGB", im.size)
    for ix in range(im.size[0]):
        for iy in range(im.size[1]):
            new_img.putpixel((ix,iy), tuple(get_pix_val(ix,iy,col) for col in ["r","g","b"]))
    return new_img


# Cloud masking function.
def mask_cloud_l8(image):

    cloudShadowBitMask = ee.Number(2).pow(3).int()
    cloudsBitMask = ee.Number(2).pow(5).int()
    qa = image.select('pixel_qa')
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
        qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask).select(bands).divide(10000)


def construct_region_string(point):
    """
    convert a list of two floats [lat, long]
    into a string representation of four sets of [lat,long]
    Assume our point is at the centre.
    """
    left = point[0] - 0.05
    right = point[0] + 0.05
    top = point[1] + 0.05
    bottom = point[1] - 0.05
    return str([[left,top],[right,top],[right,bottom],[left,bottom]])


def download_and_unzip(url, output_tmpdir):
    r = requests.get(url)
    if not r.status_code == 200:
        print(" HTTP Error!")
        return
    with open(output_file, "wb") as outfile:
        outfile.write(r.content)
    with ZipFile(output_file, 'r') as zip_obj:
        zip_obj.extractall()


def download_image(image_collection, # name
                   coords,   # (long, lat) or [(long,lat),...,...,...]
                   size, # in m
                   bands, # []
                   start_date, # 'yyyy-mm-dd'
                   end_date, # 'yyyy-mm-dd'
                   output_dir,
                   output_name,
                   region=None):
    """
    Download specified image to output directory
    """
    image_coll = ee.ImageCollection(image_collection)
    if len(coords) == 2:
      geom = ee.Geometry.Point(coords)
    else:
      geom = ee.Geometry.Rectangle(coords)
    dataset = image_coll.filterBounds(geom)\
    .filterDate(start_date, end_date)
    selectors = bands
    dataset = dataset.select(selectors)
    data = dataset.toList(dataset.size())
    print("Size of data list is {}".format(data.size().getInfo()))
    if not region:
        region = construct_region_string(coords)
    urls = []
    for i in range(data.size().getInfo()):
        image = ee.Image(data.get(i));
        image = image.select(bands)
#        region = construct_region_string(coords)

        url = image.getDownloadURL(
            {'region': region,
             'scale': size}
        )
        urls.append(url)
    return urls


def sanity_check_args(args):
    """
    Check that the user has set a self-consistent set of arguments.
    """
    if args.coords_point and args.coords_rect:
      raise RuntimeError("Need to specify ONE of --coords_point or coords_rect")
    if (args.coords_point or args.coords_rect) and args.input_coords_file:
      raise RuntimeError("Specify EITHER an input_coords_file OR coords_point or coords_rect")


def process_input_file(filename,
                       start_date,
                       end_date,
                       image_coll,
                       bands,
                       output_dir,
                       output_name):
    """
    Loop through an input file with one set of coordinates per line
    """
    if not os.path.exists(filename):
        raise RuntimeError("Input file {} does not exist".format(filename))
    infile = open(filename,"r")
    for line in infile.readlines():
        coords_all = [float(x) for x in args.coords_rect.split(",")]
    pass


def main():
    """
    use command line arguments to choose images.
    """
    parser = argparse.ArgumentParser(description="download from EE")
    parser.add_argument("--image_coll",help="image collection",
                        default="LANDSAT/LC08/C01/T1_SR")
    parser.add_argument("--start_date",help="YYYY-MM-DD",
                        default="2016-01-01")
    parser.add_argument("--end_date",help="YYYY-MM-DD",
                        default="2016-06-30")
    parser.add_argument("--coords_point",help="'lat,long'")
    parser.add_argument("--coords_rect",help="'lat1,long1,lat2,long2,...,...'")
    parser.add_argument("--bands",help="string containing comma-separated list",
                        default="B8")
    parser.add_argument("--output_dir",help="output directory",
                        default=".")
    parser.add_argument("--output_name",help="base of output filename",
                      default="gee_img")
    parser.add_argument("--input_coords_file",help="text file with coordinates, one per line")
    args = parser.parse_args()
    sanity_check_args(args)

    image_coll = args.image_coll
    start_date = args.start_date
    end_date = args.end_date
    output_dir = args.output_dir
    output_name = args.output_name
    bands = args.bands.split(",")

    if args.coords_point:
      coords = [float(x) for x in args.coords_point.split(",")]
    elif args.coords_rect:
      coords_all = [float(x) for x in args.coords_rect.split(",")]
      coords = [ [coords_all[2*i],coords_all[2*i+1]] for i in range(int(len(coords_all)/2))]
    if args.input_coords_file:
        process_input_file(args.input_coords_file,
                           image_coll,
                           start_date,
                           end_date,
                           bands,
                           output_dir,
                           output_name)


if __name__ == "__main__":
    main()
