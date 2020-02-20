"""
Functions to download and process satellite imagery from Google Earth Engine.
"""

import os
import sys
import shutil
import requests
import argparse
import dateparser
from datetime import datetime, timedelta
import numpy as np


from .gee_interface import (
    apply_mask_cloud,
    add_NDVI,
    download_and_unzip,
    get_download_urls
    )

from .image_utils import (
    convert_to_bw,
    crop_image_npix,
    save_image,
    convert_to_rgb,
    scale_tif,
    save_json,
    pillow_to_numpy,
    process_image
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


def divide_time_period_in_n_day_portions(start_date, end_date, days_per_chunk):
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
    return coords


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
        bw_ndvi = process_image(ndvi_image) # new adaptive threshold
        output_filename = construct_filename("ndvibw")
        save_image(bw_ndvi, output_dir, output_filename)
    
    # output the full-size black-and-white image
    #bw_image = convert_to_bw(merged_image, threshold)
    #output_filename = construct_filename("bw")
    #save_image(bw_image, output_dir, output_filename)


def process_coords(coords,
                   image_coll,
                   bands,
                   region_size, ## dimensions of output image in longitude/latitude
                   scale, # size of each pixel in output image (in m)
                   start_date,
                   end_date,
                   mask_cloud=False, ## EXPERIMENTAL - false by default
                   output_dir=".",
                   output_prefix='',
                   output_suffix=".png",
                   network_centrality=False,
                   sub_image_size=[50,50],
                   threshold=470):
    """
    Run through the whole process for one set of coordinates (either a point
    or a rectangle).
    """
    region = construct_region_string(coords, region_size)
    # Get download URL for all images at these coords
    download_urls = get_download_urls(coords,
                                      region,
                                      image_coll,
                                      bands,
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
            
            merged_image = convert_to_rgb(tif_filebase, bands)
            ndvi_image = scale_tif(tif_filebase, "NDVI")

            img_array = pillow_to_numpy(merged_image)
            black = [0,0,0]
            black_pix_threshold = 0.1
            n_black_pix = np.count_nonzero(np.all(img_array == black, axis=2))

            if n_black_pix / (img_array.shape[0]*img_array.shape[1]) > black_pix_threshold:
                print('Detected a low quality image, skipping to next date.')
                continue

            write_fullsize_images(tif_filebase, output_dir, output_prefix, output_suffix,
                                  coords, bands, threshold)

            # if requested, divide into smaller sub-images
            if network_centrality:
                sub_images = crop_image_npix(ndvi_image,
                                                sub_image_size[0],
                                                sub_image_size[1],
                                                region_size,
                                                coords
                )

                # loop through sub images
                for n, image in enumerate(sub_images):
                    #sub_image = convert_to_bw(image[0], threshold) # old harccoded threshold
                    sub_image = process_image(image[0]) # new adaptive threshold

                    sub_coords = image[1]
                    output_filename = os.path.basename(tif_filebase)
                    output_filename += "_{0:.3f}_{1:.3f}".format(sub_coords[0], sub_coords[1])
                    output_filename += '_' + output_suffix

                    # save sub image
                    save_image(sub_image, output_dir, output_filename)

                    # run network centrality
                    image_array = pillow_to_numpy(sub_image)
                    feature_vec, sel_pixels = subgraph_centrality(image_array)
                    feature_vec_metrics = feature_vector_metrics(feature_vec)
                    feature_vec_metrics['latitude'] = sub_coords[0]
                    feature_vec_metrics['longitude'] = sub_coords[1]
                    feature_vec_metrics['date']= output_prefix
                    output_filename = os.path.basename(tif_filebase)
                    output_filename += "_{0:.3f}_{1:.3f}".format(sub_coords[0], sub_coords[1])
                    output_filename += "_{}".format(output_prefix)
                    output_filename += '.json'
                    save_json(feature_vec_metrics, output_dir, output_filename)


def get_time_series(num_time_periods,
                    coords, # could be None if we have an input_file listing coords
                    image_coll,
                    bands,
                    region_size, ## dimensions of output image in longitude/latitude
                    scale, # size of each pixel in output image (in m)
                    start_date,
                    end_date,
                    mask_cloud=False, ## EXPERIMENTAL - false by default
                    output_dir=".",
                    output_suffix=".png", # end of output filename, including file extension
                    network_centrality = False,
                    sub_image_size=[50,50],
                    threshold=470):
    """
    Divide the time between start_date and end_date into num_time_periods periods
    and call download_images.process coords for each.
    """
    time_periods = divide_time_period(start_date, end_date, num_time_periods)
    
    for period in time_periods:
        print(f"\nProcessing the time period between {period[0]} and {period[1]}...")
        mid_period_string = find_mid_period(period[0], period[1])
        output_prefix = mid_period_string

        process_coords(coords,
                       image_coll,
                       bands,
                       region_size,
                       scale,
                       period[0],
                       period[1],
                       mask_cloud,
                       output_dir,
                       output_prefix,
                       output_suffix,
                       network_centrality,
                       threshold=threshold)

        print(f"Finihsed processing the time period between {period[0]} and {period[1]}.")
    return True
