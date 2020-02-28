"""
Functions to download and process satellite imagery from Google Earth Engine.
"""

import os
import sys
import shutil
import requests
import argparse
import dateparser
import json
from datetime import datetime, timedelta
import numpy as np
import cv2 as cv

from .gee_interface import ee_download

from .image_utils import (
    convert_to_bw,
    crop_image_npix,
    save_image,
    convert_to_rgb,
    scale_tif,
    save_json,
    pillow_to_numpy,
    process_and_threshold,
    check_image_ok
)

from .subgraph_centrality import (
    subgraph_centrality,
    feature_vector_metrics,
    generate_sc_images,
    text_file_to_array,
)

from pyveg.src.subgraph_centrality import subgraph_centrality

if os.name == "posix":
    TMPDIR = "/tmp/"
else:
    TMPDIR = "%TMP%"


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


def slice_time_period(start_date, end_date, n):
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


def get_num_n_day_slices(start_date, end_date, days_per_chunk):
    """
    Divide the full period between the start_date and end_date into n equal-length
    (to the nearest day) chunks. The size of the chunk is defined by days_per_chunk.
    Takes start_date and end_date as strings 'YYYY-MM-DD'.
    Returns an integer with the number of possible points avalaible in that time period]
    """
    start = dateparser.parse(start_date)
    end = dateparser.parse(end_date)
    if (not isinstance(start, datetime)) or (not isinstance(end, datetime)):
        raise RuntimeError("invalid time strings")
    td = end - start
    if td.days <= 0:
        raise RuntimeError("end_date must be after start_date")
    n = td.days//days_per_chunk

    return  n


def construct_image_savepath(output_dir, collection_name, coords, date_range, image_type):
    """
    Function to abstract output image filename construction. Current approach is to create
    a new dir inside `output_dir` for the satellite, and then save date and coordinate
    stamped images in this dir.
    """

    # get the mid point of the date range
    mid_period_string = find_mid_period(date_range[0], date_range[1])

    # filename is the date, coordinates, and image type
    filename = f'{mid_period_string}_{coords[0]}-{coords[1]}_{image_type}.png'

    # full path is dir + filename
    full_path = os.path.join(output_dir, filename)

    return full_path



def write_fullsize_images(tif_filebase, output_dir, output_prefix, output_suffix,
                          coords, bands, threshold):
    """
    Output black-and-white and colour, and possibly rescaled NDVI,
    images before dividing into sub-images.
    """
    def construct_filename(image_type):
        filename = os.path.basename(tif_filebase)
        filename += "_{0:.3f}_{1:.3f}".format(coords[0], coords[1])
        filename += "_10kmLargeImage_{}_{}".format(image_type,output_suffix)
        filename = output_prefix + '_' + filename
        return filename
    
    # output the full-size colour image
    merged_image = convert_to_rgb(tif_filebase, bands)
    output_filename = construct_filename("colour")
    save_image(merged_image, output_dir, output_filename)
    
    # if we have NDVI, rescale this and output it.
    if "NDVI" in bands:
        ndvi_image = scale_tif(tif_filebase, "NDVI")
        output_filename = construct_filename("ndvi")
        save_image(ndvi_image, output_dir, output_filename)
        #bw_ndvi = convert_to_bw(ndvi_image, threshold) # old method
        bw_ndvi = process_and_threshold(ndvi_image) # new adaptive threshold
        output_filename = construct_filename("ndvibw")
        save_image(bw_ndvi, output_dir, output_filename)


def run_network_centrality(output_dir, image, coords, date_range, region_size, sub_image_size=[50,50], n_sub_images=-1):
    """
    !! SVS: Suggest that this function should be moved to the subgraph_centrality.py module

    Given an input image, divide the image up into smaller sub-images
    and run network centrality on each.

    Parameters
    ----------
    image : Pillow.Image
        Full size binary thresholded input image.
    output_dir : str
        Path to save results to.
    coords : str
        Coordinates of the `image` argument. Used to calcualte
        coordinates of sub-images which are used to ID them.
    date_range : tuple of str
        (Start date, end data) for filenames. Date strings 
        must be formatted as 'YYYY-MM-DD'.
    sub_image_size : list, optional
        Defines the size of the sub-images which network
        centrality will be run on.
    n_sub_images : int, optional
        The nubmer of sub-images to process. This is useful for 
        testing and speeding up computation. Default is -1 which
        means process the entirety of the larger image.

    Returns
    ----------
    dict
        Single dictionary containing the results for each sub-image.
        Note that the results have already been written to disk as a
        json file.
    """

    date_range_midpoint = find_mid_period(date_range[0], date_range[1])
    base_filename = date_range_midpoint
    sub_image_output_dir = os.path.join(output_dir, date_range_midpoint)

    # start by dividing the image into smaller sub-images
    sub_images = crop_image_npix(image,
                                 sub_image_size[0],
                                 sub_image_size[1],
                                 region_size,
                                 coords)

    # store results
    nc_results = {}
    save_json(nc_results, output_dir, 'network_centralities.json')

    # loop through sub images
    for i, (sub_image, sub_coords) in enumerate(sub_images): # possible to parallelise?
        
        # if we already got enough results, return early
        if i >= n_sub_images and n_sub_images != -1:
            return nc_results

        # save sub image
        output_filename = f'sub{i}_'
        output_filename += "{0:.3f}-{1:.3f}".format(sub_coords[0], sub_coords[1])
        output_filename += '.png'
        save_image(sub_image, sub_image_output_dir, output_filename)

        # run network centrality
        image_array = pillow_to_numpy(sub_image)
        feature_vec, sel_pixels = subgraph_centrality(image_array)
        nc_result = feature_vector_metrics(feature_vec)
        
        nc_result['latitude'] = round(sub_coords[0], 4)
        nc_result['longitude'] = round(sub_coords[1], 4)
        nc_result['date'] = date_range_midpoint
        
        # incrementally write json file so we don't have to wait
        # for the full image to be processed before getting results
        with open(os.path.join(output_dir, 'network_centralities.json')) as json_file:
            nc_results = json.load(json_file)
            
            # keep track of the result for this sub-image
            nc_results[i] = nc_result

            # update json output
            save_json(nc_results, output_dir, 'network_centralities.json')

    return nc_results


def get_vegetation(output_dir, collection_dict, coords, date_range, region_size=0.1, scale=10, n_sub_images=-1):
    """
    Download vegetation data from Earth Engine. Save RGB, NDVI and thresholded NDVI images. If
    request, also get network centrality metrics on the thresholded NDVI image.

    Parameters
    ----------
    output_dir : str
        Where to save output images and network centrality results
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
    n_sub_images : int, optional
        The nubmer of sub-images to process. This is useful for 
        testing and speeding up computation. Default is -1 which
        means process the entirety of the larger image.

    Returns
    ----------
    dict
        If network centrality is run, we return the results in
        a dict. Other return values is `None`.
    """
    # download vegetation data for this time period
    download_path = ee_download(output_dir, collection_dict, coords, date_range, region_size, scale)

    # save the rgb image
    # ?should change this to remove the URI for the the filename (and put something in the foldername)?
    filenames = [filename for filename in os.listdir(download_path) if filename.endswith(".tif")]

    if len(filenames) == 0:
        return 

    # extract this to feed into `convert_to_rgb()`
    tif_filebase = os.path.join(download_path, filenames[0].split('.')[0])

    # save the rgb image
    rgb_image = convert_to_rgb(tif_filebase, collection_dict['RGB_bands'])

    # check image quality on the colour image
    if not check_image_ok(rgb_image):
        print('Detected a low quality image, skipping to next date.')
        return

    # if the image looks good, we can save it
    rgb_filepath = construct_image_savepath(output_dir, collection_dict['collection_name'], coords, date_range, 'RGB')
    save_image(rgb_image, os.path.dirname(rgb_filepath), os.path.basename(rgb_filepath))

    # save the NDVI image
    ndvi_image = scale_tif(tif_filebase, "NDVI")
    ndvi_filepath = construct_image_savepath(output_dir, collection_dict['collection_name'], coords, date_range, 'NDVI')
    save_image(ndvi_image, os.path.dirname(ndvi_filepath), os.path.basename(ndvi_filepath))

    # preprocess and theshold the NDVI image
    processed_ndvi = process_and_threshold(ndvi_image)
    ndvi_bw_filepath = construct_image_savepath(output_dir, collection_dict['collection_name'], coords, date_range, 'BWNDVI')
    save_image(processed_ndvi, os.path.dirname(ndvi_bw_filepath), os.path.basename(ndvi_bw_filepath))

    # run network centrality on the sub-images
    if collection_dict['do_network_centrality']:
        n_sub_images = 10 # do this for speedup while testing
        nc_output_dir = os.path.join(output_dir, 'network_centrality')
        nc_results = run_network_centrality(nc_output_dir, processed_ndvi, coords, date_range, region_size, n_sub_images=n_sub_images)

    return nc_results


def get_weather(output_dir, collection_dict, coords, date_range, region_size=0.1, scale=10):
    """
    Function to get weather data from a given image collection, coordinates and date range.
    The weather measurements are returned as a dictionary with the summary value for that region and date.
    """

    download_path = ee_download(output_dir,collection_dict, coords, date_range, region_size, scale)

    metrics_dict = {}

    for file in os.listdir(download_path):
        if file.endswith(".tif"):
            name_variable = (file.split('.'))[1]
            variable_array = cv.imread(os.path.join(download_path, file), cv.IMREAD_ANYDEPTH)

            metrics_dict[name_variable] = variable_array.mean()

    return metrics_dict


def process_single_collection(output_dir, collection_dict, coords, date_ranges, region_size=0.1, scale=10):
    """
    Process all dates for a single Earth Engine collection.
    """

    print(f'''\nProcessing collection "{collection_dict['collection_name']}".''')
    print('-'*50)

    # make a new dir inside `output_dir`
    output_subdir = os.path.join(output_dir, collection_dict['collection_name'].split('/')[0])
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)

    # store the results by date in this dict
    results = {}
    results['time-series-data'] = {}
    results['type'] = collection_dict['type']

    # for each time interval
    for date_range in date_ranges:

        print(f'Looking for data in the date range {date_range}...')
        
        # process the collection
        if collection_dict['type'] == 'vegetation':
            result = get_vegetation(output_subdir, collection_dict, coords, date_range, region_size, scale)
        else:
            result = get_weather(output_subdir, collection_dict, coords, date_range, region_size, scale)

        results['time-series-data'][find_mid_period(date_range[0], date_range[1])] = result

    print(f'''Finished processing collection "{collection_dict['collection_name']}".''')
    
    return results


def process_all_collections(output_dir, collections, coords, date_range, n_days_per_slice, region_size=0.1, scale=10):
    """
    Process all dates for all specified Earth Engine collections.
    """

    # unpack date range
    start_date, end_date = date_range

    # get the list of time intervals
    num_slices = get_num_n_day_slices(start_date, end_date, n_days_per_slice)
    date_ranges = slice_time_period(date_range[0], date_range[1], num_slices)

    # place to store results
    results_collection = {}

    for _, collection_dict in collections.items(): # possible to parallelise?

        results = process_single_collection(output_dir, collection_dict, coords, date_ranges, region_size, scale)

        results_collection[collection_dict['collection_name']] = results

    # wait for everything to finish
    print('\nSummarising results...')
    save_json(results_collection, output_dir, 'results_summary.json')
