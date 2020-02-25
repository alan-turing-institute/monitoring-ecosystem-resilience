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

Needs a relatively recent version of pillow (fork of PIL):
```
pip install --upgrade pillow
```
"""

import os

import argparse
import warnings
from pyveg.src.satellite_data_analysis import process_all_collections
from pyveg import config

def main():
    """
    use command line arguments to choose images.
    """

    example_command = """Example usage:

    python analyse_gee_data.py \\
        --start_date 2016-01-01 \\
        --end_date 2019-01-01 \\
        --coords 27.99,11.2878 \\
    """
    parser = argparse.ArgumentParser(description="download from EE", epilog=example_command, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start_date",help="YYYY-MM-DD", default="2013-03-30")
    parser.add_argument("--end_date",help="YYYY-MM-DD", default="2013-04-01")
    parser.add_argument("--coords",help="'long,lat'")

    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date

    coords = [float(x) for x in args.coords.split(",")]
    
    input_collections = config.data_collections
    print('Running with the following collections:')
    for name, collection_dict in input_collections.items():
        print(collection_dict)
    print('\n')

    process_all_collections(input_collections,
                            coords,
                            (start_date,end_date),
                            config.num_days_per_point)


    print('Done!')


if __name__ == "__main__":
    main()
