"""
Functions to help interface with GEE, in particular to download images.
"""

import os
import shutil
import requests
from datetime import datetime
from zipfile import ZipFile, BadZipFile
from geetools import cloud_mask
import cv2 as cv

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



def apply_mask_cloud(image_coll, collection_name, cloudy_pix_flag):
    """
    Different input_collections need different steps to be taken to handle
    cloudy image. The first step is to reject images that more than X%
    cloudy pixels (here X=5). The next step is to mask cloudy pixels. This
    will hopefully mean that when we take the median of the ImageCollection,
    we ignore cloudy pixels.

    Parameters
    ----------
    image_coll : ee.ImageCollection
        The ImageCollection of images from which we want to remove cloud.
    collection_name : str
        Name of the collection so that we can apply collection specific
        masking.
    cloud_pix_flag : str
        Name of the flag which details the fraction of cloudy pixels in each
        image.

    Returns
    ----------
    image_coll
        Image collection with very cloudy images removed, and masked images
        containing a tolerable amount of cloud.
    """

    # construct cloud mask if availible
    if collection_name == 'COPERNICUS/S2':
        mask_func = cloud_mask.sentinel2()
    elif collection_name == 'LANDSAT/LC08/C01/T1_SR':
        mask_func = cloud_mask.landsat8SRPixelQA()
    elif ( collection_name == 'LANDSAT/LE07/C01/T1_SR' or
           collection_name == 'LANDSAT/LT05/C01/T1_SR' or
           collection_name == 'LANDSAT/LT04/C01/T1_SR' ):
        mask_func = cloud_mask.landsat457SRPixelQA()
    else:
        print("No cloud mask logic defined for input collection {}"\
              .format(collection_name))
        return image_coll

    # images with more than this percent of cloud pixels are removed
    cloud_pix_frac = 10

    # remove images that have more than `cloud_pix_frac`% cloudy pixels
    if cloudy_pix_flag != 'None':
        image_coll = image_coll.filter(ee.Filter.lt(cloudy_pix_flag, cloud_pix_frac))

    # apply per pixel cloud mask
    image_coll = image_coll.map(mask_func)

    return image_coll


def add_NDVI(image, red_band, near_infrared_band):
    try:
        image_ndvi = image.normalizedDifference([near_infrared_band, red_band]).rename('NDVI')
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

    # GET the URL
    r = requests.get(url)
    if not r.status_code == 200:
        raise RuntimeError(" HTTP Error getting download link {}".format(url))
    # DO NOT remove output directory and recreate it
    #shutil.rmtree(output_tmpdir, ignore_errors=True)
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


def ee_prep_data(collection_dict,
                 coords,
                 date_range,
                 region_size=0.1,
                 scale=10,
                 mask_cloud=True):
    """
    Use the Earth Engine API to prepare data for download.

    Parameters
    ----------
    collection_dict : dict
        Dictionary containing information about the collection (name,
        type, bands, etc). Follows structure in the config file.
    coords : tuple of float
        (Latitude, longitude) coordinates.
    region : str
         String representation of 4 sets of [long,lat] forming rectangle
         around the point specified in coords.
    date_range : tuple of str
        (Start date, end data) for data filtering. Date strings
        must be formatted as 'YYYY-MM-DD'.
    region_size : float, optional
        Size of the output image (default is 0.1, or 1km).
    scale : int, optional
        Size of each pixel in meters (default 10).
    mask_cloud : bool, optional
        Remove cloud from images using the geetools package.

    Returns
    ----------
    list
        URLs from which we can download the data. For vegetation
        we should only get a single URL in the list, but for
        precipitation it is possible to get separate URLs for e.g.
        precipitation and weather data.
    """

    # string respresenting 4 corners of the region of interest
    region = get_region_string(coords, region_size)

    # unpack the date range
    start_date, end_date = date_range

    collection_name = collection_dict['collection_name']

    image_coll = ee.ImageCollection(collection_name)
    geom = ee.Geometry.Point(coords)

    # gather relevant images
    dataset = image_coll.filterBounds(geom).filterDate(start_date, end_date)
    dataset_size = dataset.size().getInfo()

    # check we have enough images to work with
    if dataset.size().getInfo() == 0:
        print('No images found in this date rage, skipping.')
        log_msg = 'WARN >>> No data found.'
        return [], log_msg

    # store the type of data we are working with
    data_type = collection_dict['type']

    # mask clouds in images
    if mask_cloud and data_type == 'vegetation':
        dataset = apply_mask_cloud(dataset, collection_name, collection_dict['cloudy_pix_flag'])

    # check we have enough images to work with after cloud masking
    if dataset.size().getInfo() == 0:
        print('No valid images found in this date rage, skipping.')
        log_msg = f'WARN >>> Found 0/{dataset_size} valid images after cloud filtering.'
        return [], log_msg
    else:
        print(f'Found {dataset.size().getInfo()} valid images of {dataset_size} total images in this date range.')

    image_list = []

    # if we are looking at vegetation
    if data_type == 'vegetation':

        # take the median across time of every pixel in the image
        image = dataset.median()

        # construct NDVI band from the red and near infrared bands
        image = add_NDVI(image, collection_dict['RGB_bands'][0], collection_dict['NIR_band'])

        # select only RGB + NDVI bands to download
        bands_to_select = list(collection_dict['RGB_bands']) + ['NDVI']
        #renamed_bands = [collection_dict['collection_name'].split('/')[0] + '-' + band for band in bands_to_select]
        image = image.select(bands_to_select)

        image_list.append(image)

    # for weather data
    if data_type == 'weather':

        if 'precipitation_band' in collection_dict.keys():
            # sum the precipitation across all dates
            image_weather = dataset.select(list(collection_dict['precipitation_band'])).sum()
            image_list.append(image_weather)

        if 'temperature_band' in collection_dict.keys():
            # average the temperature across all dates, may want to include
            # temperature range, min and max, in future
            image_temp = dataset.select(list(collection_dict['temperature_band'])).mean()
            image_list.append(image_temp)

    url_list =[]
    for image in image_list:
        # get a URL from which we can download the resulting data
        url = image.getDownloadURL(
            {'region': region,
            'scale': scale}
         )
        url_list.append(url)

    log_msg = f'OK   >>> Found {dataset.size().getInfo()}/{dataset_size} valid images after cloud filtering.'
    return url_list, log_msg


def get_region_string(point, size=0.1):
    """
    Construct a string of coordinates that create a box around the specified point.

    Parameters
    ----------
    point : list of float
        Latitude and longitude of the specified point.
    size : float, optional
        Side length of the output square.

    Returns
    ----------
    str
        A string with coordinates for each corner of the box. Can be passed to Earth
        Engine.
    """
    left = point[0] - size/2
    right = point[0] + size/2
    top = point[1] + size/2
    bottom = point[1] - size/2
    coords =  str([[left,top],[right,top],[right,bottom],[left,bottom]])
    return coords


def ee_download(output_dir, collection_dict, coords, date_range, region_size=0.1, scale=10):
    """
    General function to download various kinds of data from Google Earth Engine. We can get
    vegetation and weather data through this function. Cloud masking logic is performed for
    spectral data if possible.

    Parameters
    ----------
    output_dir : str
        Path to the directory where ee files will be downloaded
        and extracted to.
    collection_dict : dict
        Dictionary containing information about the collection (name,
        type, bands, etc). Follows structure in the config file.
    coords : tuple of float
        (Latitude, longitude) coordinates.
    date_range : tuple of str
        (Start date, end data) for data filtering. Date strings
        must be formatted as 'YYYY-MM-DD'.
    region_size : float, optional
        Size of the output image (default is 0.1, or 1km).
    scale : int, optional
        Size of each pixel in meters (default 10).

    Returns
    ----------
    str
        Path to download .tif files.
    """

    # get download URL for all images at these coords
    download_urls, log_msg = ee_prep_data(collection_dict,
                                 coords,
                                 date_range,
                                 region_size,
                                 scale)

    # didn't find any valid images in this date range
    if len(download_urls) == 0:
        return None, log_msg

    # path to temporary directory to download data
    sub_dir = f'gee_{coords[0]}_{coords[1]}'+"_"+collection_dict['collection_name'].replace('/', '-')
    download_dir = os.path.join(output_dir, sub_dir, date_range[0])

    # download files and unzip to temporary directory
    for download_url in download_urls:
        download_and_unzip(download_url, download_dir)

    # return the path so downloaded files can be handled by caller
    return download_dir, log_msg
