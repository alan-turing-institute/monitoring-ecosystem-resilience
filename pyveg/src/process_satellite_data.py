"""
Functions to download and process satellite imagery from Google Earth Engine.
"""

import os
import dateparser
from datetime import datetime, timedelta
import json
import numpy as np
import cv2 as cv

from multiprocessing import Pool

from .gee_interface import ee_download

from .image_utils import (
    crop_image_npix,
    convert_to_rgb,
    scale_tif,
    pillow_to_numpy,
    process_and_threshold,
    check_image_ok
)

from .file_utils import save_json, construct_image_savepath, save_image, consolidate_json_to_list
from .date_utils import find_mid_period, get_num_n_day_slices, slice_time_period

from .subgraph_centrality import (
    subgraph_centrality,
    feature_vector_metrics,
)


def process_sub_image(i, sub, sub_rgb, sub_ndvi, output_subdir, date):
    """
    function to be used by multiprocessing Pool, called for every sub-image.

    Arguments
    ---------
    i : int
       index of the sub-image
    sub: (Pillow.Image, (float,float))
       tuple containing the bwndvi sub-image, and a tuple of long,lat coords.
    sub_rgb: (Pillow.Image, (float,float))
       tuple containing the rgb sub-image, and a tuple of long,lat coords.
    sub_rgb: (Pillow.Image, (float,float))
       tuple containing the ndvi sub-image, and a tuple of long,lat coords.
    output_subdir: str
       subdirectory into which sub-image png files will be saved.

    Returns
    -------
    nc_results: list of dictionaries
               list of length n_sub_images with each entry containing a
               dictionary with coordinates, dates, feature_vec.
    """
    # sub will be a tuple (image, coords) - unpack it here
    sub_image, sub_coords = sub

    # construct sub-image filename
    output_filename = f'sub{i}_'
    output_filename += "{0:.3f}-{1:.3f}".format(sub_coords[0], sub_coords[1])
    output_filename += '.png'

    # check this sub-image passess quality control
    colour_subimage, _ = sub_rgb
    if not check_image_ok(colour_subimage):
        #print('Sub-image rejected!')
        save_image(colour_subimage, os.path.join(output_subdir, 'rejected'), output_filename)
        return

    # save accepted sub-image
    save_image(sub_image, output_subdir, output_filename)

    # average NDVI of all pixels (in case there is no veg pattern)
    ndvi_mean = round(pillow_to_numpy(sub_ndvi[0]).mean(), 4)
    ndvi_std = round(pillow_to_numpy(sub_ndvi[0]).std(), 4)

    # use the BWDVI to mask the NDVI and calculate the average
    # pixel value of veg pixels
    veg_mask = (pillow_to_numpy(sub_image) == 0)
    veg_ndvi_mean = round(pillow_to_numpy(sub_ndvi[0])[veg_mask].mean(), 4)
    veg_ndvi_std = round(pillow_to_numpy(sub_ndvi[0])[veg_mask].std(), 4)

    # run network centrality
    image_array = pillow_to_numpy(sub_image)
    feature_vec, _ = subgraph_centrality(image_array)
    nc_result = feature_vector_metrics(feature_vec)

    nc_result['longitude'] = round(sub_coords[0], 4)
    nc_result['latitude'] = round(sub_coords[1], 4)
    nc_result['date'] = date
    nc_result['feature_vec'] = list(feature_vec)
    nc_result['ndvi_mean'] = ndvi_mean
    nc_result['ndvi_std'] = ndvi_std
    nc_result['veg_ndvi_mean'] = veg_ndvi_mean
    nc_result['veg_ndvi_std'] = veg_ndvi_std

    # write json file for just this sub-image to a temporary location
    # (to be thread safe, only combine when all parallel jobs are done)
    save_json(nc_result, os.path.join(output_subdir,"tmp_json"), f"network_centrality_sub{i}.json")
    n_processed = len(os.listdir(os.path.join(output_subdir,"tmp_json")))
    print(f'Processed {n_processed} sub-images...', end='\r')


def run_network_centrality(output_dir, img_thresh, img_rgb, ndvi_img, coords, date_range, region_size, 
                           sub_image_size=[50,50], n_sub_images=-1, n_threads=4):
    """
    !! SVS: Suggest that this function should be moved to the subgraph_centrality.py module

    Given an input image, divide the image up into smaller sub-images
    and run network centrality on each.

    Parameters
    ----------
    output_dir : str
        Path to save results to.
    img_thresh : Pillow.Image
        Full size binary thresholded input NDVI image.
    img_rgb : Pillow.Image
        Full size RGB image.
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
    n_threads: int, optional
        The number of threads to use for parallel processing of sub-images.
        Default is 4.

    Returns
    ----------
    dict
        Single dictionary containing the results for each sub-image.
        Note that the results have already been written to disk as a
        json file.
    """

    date_range_midpoint = find_mid_period(date_range[0], date_range[1])
    output_subdir = os.path.join(output_dir, date_range_midpoint)

    # start by dividing the image into smaller sub-images
    sub_images = crop_image_npix(img_thresh,
                                 sub_image_size[0],
                                 sub_image_size[1],
                                 region_size,
                                 coords)

    sub_images_rgb = crop_image_npix(img_rgb,
                                     sub_image_size[0],
                                     sub_image_size[1],
                                     region_size,
                                     coords)
    
    sub_images_ndvi = crop_image_npix(ndvi_img,
                                     sub_image_size[0],
                                     sub_image_size[1],
                                     region_size,
                                     coords)

    # if requested to only look at a subset of sub-images, truncate the list here
    if n_sub_images != -1:
        sub_images = sub_images[:n_sub_images]

    # create a multiprocessing pool to handle each sub-image in parallel
    with Pool(processes=n_threads) as pool:
        # prepare the arguments for the process_sub_image function
        arguments=[(i, sub, sub_images_rgb[i], sub_images_ndvi[i], output_subdir, date_range_midpoint) \
                   for i,sub in enumerate(sub_images)]
        pool.starmap(process_sub_image, arguments)

    #nc_results = consolidate_subimage_json(output_subdir) # pre pipline way

    # re-combine the results from all sub-images
    nc_results = consolidate_json_to_list(output_subdir,
                                          output_subdir,
                                          "network_centralities.json")

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
        (Longitude, latitude) coordinates.
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
    download_path, log_msg = ee_download(output_dir, collection_dict, coords, date_range, region_size, scale)

    # save the rgb image
    filenames = [filename for filename in os.listdir(download_path) if filename.endswith(".tif")]

    if len(filenames) == 0:
        with open(os.path.join(output_dir, 'download.log'), 'a+') as file:
            file.write(f'daterange={date_range} coords={coords} >>> {log_msg}\n')
        return

    # extract this to feed into `convert_to_rgb()`
    tif_filebase = os.path.join(download_path, filenames[0].split('.')[0])

    # save the rgb image
    rgb_image = convert_to_rgb(tif_filebase, collection_dict['RGB_bands'])


    # check that RGB image isn't entirely black
    if not check_image_ok(rgb_image, 1.0):
        return None

    # logging
    with open(os.path.join(output_dir, 'download.log'), 'a+') as file:
        file.write(f'daterange={date_range} coords={coords} >>> {log_msg}\n')

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
        print('Running network centrality...')
        #n_sub_images = 20 # do this for speedup while testing
        nc_output_dir = os.path.join(output_dir, 'network_centrality')
        nc_results = run_network_centrality(nc_output_dir, processed_ndvi, rgb_image, ndvi_image, coords,
                                            date_range, region_size, n_sub_images=n_sub_images)
        print('\nDone.')
        return nc_results


def get_weather(output_dir, collection_dict, coords, date_range, region_size=0.1, scale=10):
    """
    Function to get weather data from a given image collection, coordinates and date range.
    The weather measurements are returned as a dictionary with the summary value for that region and date.
    """

    download_path, _ = ee_download(output_dir,collection_dict, coords, date_range, region_size, scale)

    metrics_dict = {}

    for file in os.listdir(download_path):
        if file.endswith(".tif"):
            name_variable = (file.split('.'))[1]
            variable_array = cv.imread(os.path.join(download_path, file), cv.IMREAD_ANYDEPTH)

            metrics_dict[name_variable] = variable_array.mean().astype(np.float64)

    return metrics_dict


def process_single_collection(output_dir, collection_dict, coords, date_ranges, region_size=0.1, scale=10):
    """
    Process all dates for a single Earth Engine collection.
    """

    print(f'''\nProcessing collection "{collection_dict['collection_name']}".''')
    print('-'*50)

    # make a new dir inside `output_dir`
    output_subdir = os.path.join(output_dir, collection_dict['collection_name'].replace('/', '-'))
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)

    # store the results by date in this dict
    results = {}
    results['type'] = collection_dict['type']
    results['time-series-data'] = {}

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


def process_all_collections(output_dir, collections, coords, date_range, region_size=0.1, scale=10):
    """
    Process all dates for all specified Earth Engine collections.
    """

    # unpack date range
    start_date, end_date = date_range

    # place to store results
    results_collection = {}

    for _, collection_dict in collections.items(): # possible to parallelise?

        # get the list of time intervals
        num_days_per_point = collection_dict['num_days_per_point']

        # pre pipeline
        #num_slices = get_num_n_day_slices(start_date, end_date, num_days_per_point)
        #date_ranges = slice_time_period(date_range[0], date_range[1], num_slices) # pass the number of slices

        # more advanced date slicing method
        date_ranges = slice_time_period(start_date,
                                        end_date,
                                        num_days_per_point+'d') # pass directly the time interval
    

        # get the data
        results = process_single_collection(output_dir, collection_dict, coords, date_ranges, region_size, scale)

        # save an intermediate file for each collection in case of crash
        pathsafe_collection_name = collection_dict['collection_name'].replace('/', '-')
        output_subdir = os.path.join(output_dir, pathsafe_collection_name)
        save_json(results, output_subdir, pathsafe_collection_name+'_results.json')

        # store in final dict
        results_collection[collection_dict['collection_name']] = results

    # wait for everything to finish
    print('\nSummarising results...')
    save_json(results_collection, output_dir, 'results_summary.json')
