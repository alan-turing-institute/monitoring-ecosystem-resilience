## Google Earth Engine related tools

[Google Earth Engine](https://earthengine.google.com) is a powerful tool for obtaining and analysing satellite imagery.
This directory contains some useful scripts for interacting with the Earth Engine API.

To install the python API, on a recent version (>=3.6) of python, do:
```
pip install earthengine-api
```
or, if you don't use pip and/or would like to install from source, follow the instructions [here](https://developers.google.com/earth-engine/python_install_manual).

## Installing the package

From the main `monitoring-ecosystem-resilience` directory (one above this one), do
```
pip install .
```

Before running the any of the GEE-related scripts, from the command-line, do:
```
 earthengine authenticate
```
and follow the instructions there (a browser tab will open with a code for you to copy/paste back into the command line prompt).

### Downloading and analysing images from GEE:

For a list of command line arguments do
```
pyveg_gee_analysis --help
```

### Examples:
```
pyveg_gee_analysis --image_coll COPERNICUS/S2 --start_date 2016-01-01 --end_date 2016-06-30 --coords 27.95,11.57 --bands B2,B3,B4 --region_size 0.1 --output_dir /tmp/TEST_IMAGES
```

Note that by default the image downloaded from GEE will be split up into 50x50 pixel images, which will also be
converted into monochrome based on the sum of r,g,b pixel values.

### Getting a time series of images

The ```start_date``` and ```end_date``` can be used in conjunction with the ```--num_time_points``` argument to divide the time period into this number of (approx) equal length periods, and download images for all of them.

```
pyveg_gee_analysis --image_coll COPERNICUS/S2 --start_date 2016-01-01 --end_date 2017-01-01 --coords 27.95,11.57 --bands B2,B3,B4 --region_size 0.1 --output_dir /tmp/TEST_IMAGES --num_time_points 12
```


## Pattern simulation


The ```generate_patterns.py``` functionality originates from some Matlab code by Stefan Dekker, Willem Bouten, Maarten Boerlijst and Max Rietkerk (included in the "matlab" directory), implementing the scheme described in:

Rietkerk et al. 2002. Self-organization of vegetation in arid ecosystems. The American Naturalist 160(4): 524-530.

To run this simulation in Python, there is an entrypoint defined.  Type:
```
pyveg_gen_pattern --help
```
to see the options.  The most useful option is the `--rainfall` parameter which sets a parameter (the rainfall in mm) of the simulation - values between 1.2 and 1.5 seem to give rise to a good range of patterns.
Other optional parameters for `generate_patterns.py` allow the generated image to be output as a csv file or a png image.  The `--transpose` option rotates the image 90 degrees (this was useful for comparing the Python and Matlab network-modelling code).
Other parameters for running the simulation are in the file `patter_gen_config.py`, you are free to change them.


## Network centrality

There is an entrypoint defined in `setup.py` that runs the *main* function of `calc_euler_characteristic.py`:
```
pyveg_calc_EC --help
```
will show the options.

* `--input_txt` allows you to give the input image as a csv, with one row per row of pixels.  Inputs are expected to be "binary", only containing two possible pixel values (typically 0 for black and 255 for white).
* `--input_img` allows you to pass an input image (png or tif work OK).  Note again that input images are expected to be "binary", i.e. only have two colours.
* `--sig_threshold` (default value 255) is the value above (or below) which a pixel is counted as signal (or background)
* `--upper_threshold` determines whether the threshold above is an upper or lower threshold (default is to have a lower threshold - pixels are counted as "signal" if their value is greater-than-or-equal-to the threshold value).
* `--use_diagonal_neighbours` when calculating the adjacency matrix, the default is to use "4-neighbours" (i.e. pixels immediately above, below, left, or right).  Setting this option will lead to "8-neighbours" (i.e. those diagonally adjacent) to also be included.
* `--num_quantiles` determines how many elements the output feature vector will have.
* `--do_EC` Calculate the Euler Characteristic to fill the feature vector.  Currently this is required, as the alternative approach (looking at the number of connected components) is not fully debugged.

Note that if you use the `-i` flag when running python, you will end up in an interactive python session, and have access to the `feature_vec`, `sel_pixels` and `sc_images` variables.

Examples:
```
pyveg_calc_EC --input_txt ../binary_image.txt --do_EC
>>> sc_images[50].show() # plot the image with the top 50% of pixels (ordered by subgraph centrality) highlighted.
>>> plt.plot(list(sel_pixels.keys()), feature_vec, "bo") # plot the feature vector vs pixel rank
>>> plt.show()
```
