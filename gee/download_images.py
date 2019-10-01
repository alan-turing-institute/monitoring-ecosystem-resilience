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

"""

import os
import sys
import shutil
import requests
import argparse
from zipfile import ZipFile

import ee
ee.Initialize()

from image_utils import *

if os.name == "posix":
    TMPDIR = "/tmp/"
else:
    TMPDIR = "%TMP%"


def save_image(image, output_dir, output_filename):
    """
    Given a PIL.Image (list of pixel values), save
    to requested filename - note that the file extension
    will determine the output file type, can be .png, .tif,
    probably others...
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    image.save(output_path)
    print("Saved image {}".format(output_path))




# Cloud masking function.
def mask_cloud(image, input_coll, bands):
    """
    Different input_collections need different steps to be taken to filter
    out cloud.
    """
    if "LANDSAT" in input_coll:
        cloudShadowBitMask = ee.Number(2).pow(3).int()
        cloudsBitMask = ee.Number(2).pow(5).int()
        qa = image.select('pixel_qa')
        mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
            qa.bitwiseAnd(cloudsBitMask).eq(0))
        return image.updateMask(mask).select(bands).divide(10000)
    elif "COPERNICUS" in input_coll:
        image = image.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20))
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        qa = image.select("QA60")


def construct_region_string(point, size=0.1):
    """
    convert a list of two floats [long, lat]
    into a string representation of four sets of [long,lat]
    Assume our point is at the centre.
    """
    left = point[0] - size/2
    right = point[0] + size/2
    top = point[1] + size/2
    bottom = point[1] - size/2
    coords =  str([[left,top],[right,top],[right,bottom],[left,bottom]])
    print(coords)
    return coords

def download_and_unzip(url, output_tmpdir):
    """
    Given a URL from GEE, download it (will be a zipfile) to
    a temporary directory, then extract archive to that same dir.
    Then find the base filename of the resulting .tif files (there
    should be one-file-per-band) and return that.
    """
    filebases = []
    # GET the URL
    r = requests.get(url)
    if not r.status_code == 200:
        raise RuntimeError(" HTTP Error getting download link {}".format(url))
    # remove output directory and recreate it
    shutil.rmtree(output_tmpdir, ignore_errors=True)
    os.makedirs(output_tmpdir, exist_ok=True)
    output_zipfile = os.path.join(output_tmpdir,"gee.zip")
    with open(output_zipfile, "wb") as outfile:
        outfile.write(r.content)
    with ZipFile(output_zipfile, 'r') as zip_obj:
        zip_obj.extractall(path=output_tmpdir)

    tif_files = [filename for filename in os.listdir(output_tmpdir) \
                 if filename.endswith(".tif")]
    if len(tif_files) == 0:
        raise RuntimeError("No files extracted")
    # get the filename before the "Bx" band identifier
    tif_filebases = [tif_file.split(".")[0] for tif_file in tif_files]
    # get the unique list
    tif_filebases = set(tif_filebases)
    # prepend the directory name to each of the filebases
    return [os.path.join(output_tmpdir, tif_filebase) \
            for tif_filebase in tif_filebases]


def get_download_urls(coords,   # (long, lat) or [(long,lat),...,...,...]
                      image_collection, # name
                      bands, # []
                      region_size, # size of output image region in long/lat
                      scale, # output pixel size in m
                      start_date, # 'yyyy-mm-dd'
                      end_date, # 'yyyy-mm-dd'
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
#    dataset = mask_cloud(dataset, image_collection, bands)
    image = dataset.median()

    image = image.select(bands)
#    data = dataset.toList(dataset.size())
    if not region:
        region = construct_region_string(coords, region_size)
    urls = []

    url = image.getDownloadURL(
        {'region': region,
         'scale': scale}
    )
    urls.append(url)
    print("Found {} sets of images for coords {}".format(len(urls),coords))
    return urls


def process_coords(coords,
                   image_coll,
                   bands,
                   region_size, ## dimensions of output image in longitude/latitude
                   scale, # size of each pixel in output image (in m)
                   start_date,
                   end_date,
                   output_dir,
                   output_suffix):
    """
    Run through the whole process for one set of coordinates (either a point
    or a rectangle).
    """
    # Get download URL for all images at these coords
    download_urls = get_download_urls(coords,
                                      image_coll,
                                      bands,
                                      region_size,
                                      scale,
                                      start_date,
                                      end_date)

    # loop through these URLS, download zip files, and combine tif files
    # for each band into RGB output images.
    for i, url in enumerate(download_urls):
        # construct a temp directory name based on coords and index
        # of this url in the list
        tmpdir = os.path.join(TMPDIR, "gee_"+str(coords[0])+"_"\
                              +str(coords[1])+"_"+str(i))
        tif_filebases = download_and_unzip(url,tmpdir)
        # Now should have lots of .tif files in a temp dir - merge them
        # into RGB image files in our chosen output directory
        for tif_filebase in tif_filebases:
            merged_image = combine_tif(tif_filebase, bands)
            # now save this
            output_filename = tif_filebase.split("/")[-1]
            output_filename += "_{}_{}".format(coords[0],coords[1])
            output_filename += output_suffix
            save_image(merged_image, output_dir, output_filename)


def process_input_file(filename,
                       image_coll,
                       bands,
                       region_size,
                       scale,
                       start_date,
                       end_date,
                       output_dir,
                       output_suffix):
    """
    Loop through an input file with one set of coordinates per line
    """
    if not os.path.exists(filename):
        raise RuntimeError("Input file {} does not exist".format(filename))
    infile = open(filename,"r")
    for line in infile.readlines():
        coords = [float(x) for x in line.strip().split(",")]
        print("Processing {}".format(coords))
        process_coords(coords,
                       image_coll,
                       bands,
                       region_size,
                       scale,
                       start_date,
                       end_date,
                       output_dir,
                       output_suffix)


def sanity_check_args(args):
    """
    Check that the user has set a self-consistent set of arguments.
    """
    if args.coords_point and args.coords_rect:
      raise RuntimeError("Need to specify ONE of --coords_point or coords_rect")
    if (args.coords_point or args.coords_rect) and args.input_file:
      raise RuntimeError("Specify EITHER an input_file OR coords_point or coords_rect")



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

    if args.coords_point:
      coords = [float(x) for x in args.coords_point.split(",")]
    elif args.coords_rect:
      coords_all = [float(x) for x in args.coords_rect.split(",")]
      coords = [ [coords_all[2*i],coords_all[2*i+1]] for i in range(int(len(coords_all)/2))]
    if args.input_file:
        process_input_file(args.input_file,
                           image_coll,
                           bands,
                           region_size,
                           scale,
                           start_date,
                           end_date,
                           output_dir,
                           output_suffix)
    else:
        # individual set of coordinates
        process_coords(coords,
                       image_coll,
                       bands,
                       region_size,
                       scale,
                       start_date,
                       end_date,
                       output_dir,
                       output_suffix)


if __name__ == "__main__":
    main()
