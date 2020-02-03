"""
Functions to help interface with GEE, in particular to download images.
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

from .image_utils import (
    convert_to_bw,
    crop_image_npix,
    save_image,
    combine_tif
)

if os.name == "posix":
    TMPDIR = "/tmp/"
else:
    TMPDIR = "%TMP%"

LOGFILE = os.path.join(TMPDIR, "failed_downloads.log")




# EXPERIMENTAL Cloud masking function.  To be applied to Images (not ImageCollections)
def apply_mask_cloud(image, input_coll):
    """
    Different input_collections need different steps to be taken to filter
    out cloud.
    """
    if input_coll=='LANDSAT/LC08/C01/T1_SR':
        mask_func = cloud_mask.landsat8SRPixelQA()
        image = image.map(mask_func)
        return image

    elif input_coll=='COPERNICUS/S2':
        mask_func = cloud_mask.sentinel2()
        image = image.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20)).map(mask_func)
        return image
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


def get_download_urls(coords, # [long,lat]
                      region,   # string representation of 4 sets of [long,lat] forming rectangle around coords
                      image_collection, # name
                      bands, # []
                      scale, # output pixel size in m
                      start_date, # 'yyyy-mm-dd'
                      end_date, # 'yyyy-mm-dd'
                      mask_cloud=True):
    """
    Download specified image to output directory
    """
    image_coll = ee.ImageCollection(image_collection)
    geom = ee.Geometry.Point(coords)

    dataset = image_coll.filterBounds(geom)\
    .filterDate(start_date, end_date)

    if mask_cloud:
        dataset = apply_mask_cloud(dataset, image_collection)

    if dataset.size().getInfo() == 0:
        print('No valid images found in this date rage, skipping.')
        return []

    image = dataset.median()

    if 'NDVI' in bands:
        image = add_NDVI(image)

    image = image.select(bands)

    urls = []

    url = image.getDownloadURL(
        {'region': region,
         'scale': scale}
    )
    urls.append(url)
    print("Found {} sets of images for coords {}".format(len(urls),coords))
    return urls
