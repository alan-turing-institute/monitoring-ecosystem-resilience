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
python download_images.py --image_coll COPERNICUS/S2 --start_date 2016-01-01 --end_date 2016-06-30 --coords_point 27.95,11.57 --bands B2,B3,B4 --region_size 0.1
python download_images.py --image_coll COPERNICUS/S2 --start_date 2016-01-01 --end_date 2016-06-30 --input_file coordinates.txt --bands B2,B3,B4 --region_size 0.1
```

