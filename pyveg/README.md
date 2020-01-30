# Google Earth Engine related tools

[Google Earth Engine](https://earthengine.google.com) is a powerful tool for obtaining and analysing satellite imagery.
This directory contains some useful scripts for interacting with the Earth Engine API.

To install the python API, on a recent version (>=3.6) of python, do:
```
pip install earthengine-api
```
or, if you don't use pip and/or would like to install from source, follow the instructions [here](https://developers.google.com/earth-engine/python_install_manual).

## *download_images.py*
This script can download images from google earth engine to local disk.

Dependencies:
```
requests
zipfile
pillow
earthengine-api
```


Before running the script, from the command-line, do:
```
 earthengine authenticate
```
and follow the instructions there (a browser tab will open with a code for you to copy/paste back into the command line prompt).

For a list of command line arguments do
```
python download_images.py --help
```

### Examples:
```
python download_images.py --image_coll COPERNICUS/S2 --start_date 2016-01-01 --end_date 2016-06-30 --coords_point 27.95,11.57 --bands B2,B3,B4 --region_size 0.1 --output_dir /tmp/TEST_IMAGES
python download_images.py --image_coll COPERNICUS/S2 --start_date 2016-01-01 --end_date 2016-06-30 --input_file coordinates.txt --bands B2,B3,B4 --region_size 0.1 --output_dir /tmp/TEST_IMAGES
```

Note that by default the image downloaded from GEE will be split up into 50x50 pixel images, which will also be
converted into monochrome based on the sum of r,g,b pixel values.

### Getting a time series of images

The ```start_date``` and ```end_date``` can be used in conjunction with the ```--num_time_points``` argument to divide the time period into this number of (approx) equal length periods, and download images for all of them.

```
python download_images.py --image_coll COPERNICUS/S2 --start_date 2016-01-01 --end_date 2017-01-01 --coords_point 27.95,11.57 --bands B2,B3,B4 --region_size 0.1 --output_dir /tmp/TEST_IMAGES --num_time_points 12
```

### Also

Note that we use the GEE convention for coordinates, i.e. `(longitude,latitude)`.
