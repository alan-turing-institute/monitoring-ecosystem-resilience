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
import pandas as pd

import ee
ee.Initialize()


def get_angles_times(image_collection):
    """
    Loop through all images in an image collection and get some numbers out
    """
    azimuths = []
    zeniths = []
    timestamps = []
    for i in range(image_collection.size().getInfo()):
        if i%10 == 0:
            print("Processing image {}".format(i))
        metadata = image_collection.getInfo()['features'][i]
        azimuth = metadata["properties"]['MEAN_SOLAR_AZIMUTH_ANGLE']
        zenith = metadata["properties"]['MEAN_SOLAR_ZENITH_ANGLE']
        timestamp = metadata["properties"]['GENERATION_TIME']
        azimuths.append(azimuth)
        zeniths.append(zenith)
        timestamps.append(timestamp)
    return azimuths, zeniths, timestamps



def filter_image_collection(coords, # [long,lat]
                            image_collection, # name
                            start_date, # 'yyyy-mm-dd'
                            end_date, # 'yyyy-mm-dd'
                            ):
    """
    Download specified image to output directory
    """
    image_coll = ee.ImageCollection(image_collection)
    geom = ee.Geometry.Point(coords)

    dataset = image_coll.filterBounds(geom)\
    .filterDate(start_date, end_date)
    print("Size of image_coll is {}".format(dataset.size().getInfo()))
    return dataset


def make_dataframe(timestamps,azimuths,zeniths):
    """
    convert to a pandas dataframe with the time as the index
    """
    dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
    df = pd.DataFrame({"date": dates,
                       "azimuthal_angle": azimuths,
                       "zenith_angle": zeniths})
    return df


def main():
    parser = argparse.ArgumentParser(description="solar angle vs time")
    parser.add_argument("--image_collection",help="image collection", default="COPERNICUS/S2")
    parser.add_argument("--coords",help="comma-separated long,lat")
    parser.add_argument("--start_date",help="YYYY-MM-DD")
    parser.add_argument("--end_date",help="YYYY-MM-DD")
    args = parser.parse_args()
    coords = [float(x) for x in args.coords.split(",")]
    start_date = args.start_date
    end_date = args.end_date
    image_collection = args.image_collection
    dataset = filter_image_collection(coords,
                                      image_collection,
                                      start_date,end_date)
    az, zen, ts = get_angles_times(dataset)
    df = make_dataframe(ts, az, zen)


if __name__ == "__main__":
    df = main()
