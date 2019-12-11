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
a list of coordinates (one per line in format:   long,lat ) OR receive an
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
import sys
import shutil
import requests
import argparse
import dateparser
from datetime import datetime, timedelta
from zipfile import ZipFile, BadZipFile
from geetools import cloud_mask


import ee
ee.Initialize()

from image_utils import convert_to_bw, crop_image_npix, save_image, combine_tif

if os.name == "posix":
    TMPDIR = "/tmp/"
else:
    TMPDIR = "%TMP%"

LOGFILE = os.path.join(TMPDIR, "failed_downloads.log")


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


# EXPERIMENTAL Cloud masking function.  To be applied to Images (not ImageCollections)
def mask_cloud(image, input_coll):
    """
    Different input_collections need different steps to be taken to filter
    out cloud.
    """
    if "LANDSAT" in input_coll:
        mask_func = cloud_mask.landsat8ToaBQA()
        return mask_func(image)

    elif "COPERNICUS" in input_coll:
        image = image.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20))
        mask_func = cloud_mask.sentinel2()
        return mask_func(image)
    else:
        print("No cloud mask logic defined for input collection {}"\
              .format(input_coll))
        return image


def add_NDVI(image):
    try:
        nir = image.select('B5');
        red = image.select('B4');
#        image_ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
        image_ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
        return ee.Image(image).addBands(image_ndvi)
    except:
        print ("Something went wrong in the NDVI variable construction")
        return image


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
    ## catch zipfile-related exceptions here, and if they arise,
    ## write the name of the zipfile and the url to a logfile
    try:
        with ZipFile(output_zipfile, 'r') as zip_obj:
            zip_obj.extractall(path=output_tmpdir)
    except(BadZipFile):
        with open(LOGFILE, "a") as logfile:
            logfile.write("{}: {} {}\n".format(str(datetime.now()),
                                               output_zipfile,
                                               url))
            return None
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
                      region=None,
                      mask_cloud=False):
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

    image = dataset.median()
    if mask_cloud:
        image = mask_cloud(image, image_collection)
    if 'NDVI' in bands:
        image = add_NDVI(image)

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
                   mask_cloud=False, ## EXPERIMENTAL - false by default
                   output_dir=".",
                   output_suffix="_gee.png",
                   divide_images=True,
                   sub_image_size=[50,50]):
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
                                      end_date,
                                      mask_cloud)

    # loop through these URLS, download zip files, and combine tif files
    # for each band into RGB output images.
    for i, url in enumerate(download_urls):
        # construct a temp directory name based on coords and index
        # of this url in the list
        tmpdir = os.path.join(TMPDIR, "gee_"+str(coords[0])+"_"\
                              +str(coords[1])+"_"+str(i))
        tif_filebases = download_and_unzip(url,tmpdir)
        if not tif_filebases:
            continue
        # Now should have lots of .tif files in a temp dir - merge them
        # into RGB image files in our chosen output directory
        for tif_filebase in tif_filebases:
            merged_image = combine_tif(tif_filebase, bands)
            output_filename = os.path.basename(tif_filebase)
            output_filename += "_{0:.3f}_{1:.3f}".format(coords[0], coords[1])
            output_filename += "_{}".format(output_suffix)
            ## if requested, divide into smaller sub-images
            if divide_images:
                sub_images = crop_image_npix(merged_image,
                                             sub_image_size[0],
                                             sub_image_size[1],
                                             region_size,
                                             coords
                )
                # now save these
                for n, image in enumerate(sub_images):
                    sub_image = convert_to_bw(image[0],threshold=470)
                    sub_coords = image[1]
                    output_filename = os.path.basename(tif_filebase)
                    output_filename += "_{0:.3f}_{1:.3f}".format(sub_coords[0], sub_coords[1])
                    output_filename += output_suffix
                    save_image(sub_image, output_dir, output_filename)
            else:
                ## Save the full-size image
                save_image(merged_image, output_dir, output_filename)
        return


def process_input_file(filename,
                       image_coll,
                       bands,
                       region_size,
                       scale,
                       start_date,
                       end_date,
                       mask_cloud=False,
                       output_dir=".",
                       output_suffix="gee"):
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
                       mask_cloud,
                       output_dir,
                       output_suffix)


def find_mid_period(start_time, end_time):
    """
    Given two strings in the format YYYY-MM-DD return a
    string in the same format representing the middle (to
    the nearest day)
    """
    t0 = dateparser.parse(start_time)
    t1 = dateparser.parse(end_time)
    td = (t1 - t0).days
    mid = (t0 + timedelta(days=(td//2))).isoformat()
    return mid.split("T")[0]



def get_time_series(num_time_periods,
                    input_file, # could be None if we are looking at individual coords
                    coords, # could be None if we have an input_file listing coords
                    image_coll,
                    bands,
                    region_size, ## dimensions of output image in longitude/latitude
                    scale, # size of each pixel in output image (in m)
                    start_date,
                    end_date,
                    mask_cloud=False, ## EXPERIMENTAL - false by default
                    output_dir=".",
                    divide_images=True,
                    sub_image_size=[50,50]):
    """
    Divide the time between start_date and end_date into num_time_periods periods
    and call download_images.process coords for each.
    """
    time_periods = divide_time_period(start_date, end_date, num_time_periods)
    for period in time_periods:
        print("Processing the time period between {} and {}".format(period[0],period[1]))
        mid_period_string = find_mid_period(period[0], period[1])
        output_suffix = "_{}.png".format(mid_period_string)
        if input_file:
            process_input_file(input_file,
                               image_coll,
                               bands,
                               region_size,
                               scale,
                               period[0],
                               period[1],
                               mask_cloud,
                               output_dir,
                               output_suffix)
        else:
            process_coords(coords,
                           image_coll,
                           bands,
                           region_size,
                           scale,
                           period[0],
                           period[1],
                           mask_cloud,
                           output_dir,
                           output_suffix)
        print("Finished processing {}".format(mid_period_string))
    return True


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
    parser.add_argument("--input_file",help="text file with coordinates, one per line")
    parser.add_argument("--mask_cloud",help="EXPERIMENTAL - apply cloud masking function",action='store_true')

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
