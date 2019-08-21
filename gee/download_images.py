#!/usr/bin/env python

"""
Script to download images from Google Earth Engine
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
                   coords,   # (long, lat)
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
