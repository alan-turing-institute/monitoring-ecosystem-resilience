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

import ee
ee.Initialize()


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


def download_and_unzip(url, output_file):
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
                   region=None):
    """
    Download specified image to output directory
    """
    image_coll = ee.ImageCollection(image_collection)
#    point = ee.Geometry.Point(coords)
    rect = ee.Geometry.Rectangle(coords)
    dataset = image_coll.filterBounds(rect)\
    .filterDate(start_date, end_date)
    selectors = bands
    dataset = dataset.select(selectors)
    data = dataset.toList(dataset.size())
    print("Size of data list is {}".format(data.size().getInfo()))
    if not region:
        region = construct_region_string(coords)

    for i in range(data.size().getInfo()):
        image = ee.Image(data.get(i));
        image = image.select(bands)
#        region = construct_region_string(coords)

        url = image.getDownloadURL(
            {'region': region,
             'scale': 10}
        )
        return url


def main():
    """
    use command line arguments to choose images.
    """
    parser = argparse.ArgumentParser(description="download from EE")
    parser.add_argument("--image_coll",help="image collection",
                        default="LANDSAT/LC08/C01/T1_SR")
    parser.add_argument("--start_date",help="YYYY-MM-DD",
                        default="2014-02-10")
    parser.add_argument("--end_date",help="YYYY-MM-DD",
                        default="2014-02-20")
    parser.add_argument("--coords_point",help="'lat,long'")
    parser.add_argument("--coords_rect",help="'lat1,long1,lat2,long2,...,...'")
    parser.add_argument("--bands",help="string containing comma-separated list",
                        default="B8")
    parser.add_argument("--output_dir",help="output directory",
                        default=".")
    parser.add_argument("--output_name",help="base of output filename",
                      default="gee_img")
    args = parser.parse_args()
    if args.coords_point and args.coords_rect:
      raise RuntimeError("Need to specify ONE of --coords_point or coords_rect")
    if args.coords_point:
      coords = [float(x) for x in args.coords_point.split(",")]
    elif args.coords_rect:
      coords_all = [float(x) for x in args.coords_rect.split(",")]
      coords = [ [coords_all[2*i],coords_all[2*i+1]] for i in range(int(len(coords_all)/2))]
    print("coords are {}".format(coords))


if __name__ == "__main__":
    main()
