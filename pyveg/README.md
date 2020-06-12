# The `pyveg` Package

This page contains an installation guide, and some usage examples for this package.

## Installation

`pyveg` requires Python 3.6 or greater. To install, start by creating a fresh `conda` environment.
```
conda create -n veg python=3.7
conda activate veg
```
Get the source.
```
git clone git@github.com:alan-turing-institute/monitoring-ecosystem-resilience.git
```
Enter the repository and check out a relevant branch if necessary (the `develop` branch contains the most
up to date stable version of the code).
```
cd monitoring-ecosystem-resilience
git checkout develop
```
Install the package using `pip`.
```
pip install .
```
If you plan on making changes to the source code, you can instead run `pip install -e .`.

Before using the Google Earth Engine API, you need to sign up with a Google account 
[here](https://earthengine.google.com/new_signup/), and authenticate.  To authenticate, run
```
earthengine authenticate
```
A new browser window will open. Copy the token from this window to the terminal prompt to 
complete the authentication process.


## Google Earth Engine

[Google Earth Engine](https://earthengine.google.com) (GEE) is a powerful tool for obtaining 
and analysing satellite imagery. This directory contains some useful scripts for interacting 
with the Earth Engine API. The earth engine API is installed automatically as part of the 
`pyveg` package installation. If you wish to install it separately, you can follow the 
instructions [here](https://developers.google.com/earth-engine/python_install_manual).


## Downloading data from GEE

To run a `pyveg` download job, use
```
pyveg_run_pipeline --config_file <path to config>
```

The download job is fully specified by a configuration file, which you point
to using the `--config_file` argument. A sample config file is found at 
`pyveg/configs/config_all.py`. You can also optionally specify a string
to identify the download job using the `--name` argument.

Note that we use the GEE convention for coordinates, i.e. `(longitude,latitude)`.


### Download configuration

Inside the config file, you can specify the output directory for the data 
downloaded, start and end dates for the download job,
as well as the location, and GEE collections to download from. The supported 
GEE collections are themselves specified in `pyveg/configs/collections.py`. 

Collections can either be of type "vegetation" or "weather".
Each type has its own config options, and you can find out more by 
taking a look inside the `collections.py` file:
    
**Vegetation Collection Example**:
```        
'Landsat8' : {
     'collection_name': 'LANDSAT/LC08/C01/T1_SR',
     'data_type': 'vegetation',
     'RGB_bands': ['B4','B3','B2'],
     'NIR_band': 'B5',
     'cloudy_pix_flag': 'CLOUD_COVER',
     'min_date': '2013-01-01',
     'max_date': time.strftime("%Y-%m-%d"),
     'time_per_point': "1m"
}
```
    
**Weather Collection Example**:
```
'ERA5' : {
     'collection_name': 'ECMWF/ERA5/MONTHLY',
     'data_type': 'weather',
     'precipitation_band': ['total_precipitation'],
     'temperature_band': ['mean_2m_air_temperature'],
     'min_date': '1979-01-01',
     'max_date': time.strftime("%Y-%m-%d"),
     'time_per_point': "1m"
} 
```

### More Details on Downloading

During the download job, `pyveg` will break up your specified date range into a time series, and
download data at each point in the series. Note that by default the vegetation images downloaded
from GEE will be split up into 50x50 pixel images, vegetation metrics are then calculated on the 
sub-image level. Both colour (RGB) and Normalised Difference Vegetation Index (NDVI) images are 
downloaded and stored. Vegetation metrics include the mean NDVI pixel intensity across sub-images,
and also network centrality metrics, discussed in more detail below.

For weather collections e.g. the ERA5, due to coarser resolution, the precipitation and temperature 
"images" are averaged into a single value at each point in the time series.


## Analysing the Download Data

Once you have downloaded the data from GEE, the `pyveg_gee_analysis` command
allows you to process and analyse the output. To run:
```
pyveg_gee_analysis --input_dir <path_to_pyveg_download_output_dir>
```
The analysis code preprocesses the data and produces a number of plots. These 
will be saved in an `analysis/` subdirectory inside the `<path_to_pyveg_download_output_dir>`
directory..


### Preprocessing

`pyevg` supports the following preprocessing operations:
- Identify and remove outliers from the time series.
- Fill missing values in the time series (based on a seasonal average),
  or resample the time series using linear interpolation between points.
- Smoothing of the time series using a LOESS smoother.
- Calculation of residuals between the raw and smoothed time series.
- Deseasonalising (using first differencing), and detrending using STL.

### Plots

In the `analysis/` subdirectory, `pyveg` creates the following plots:
- Time series plots containing vegetation and precipitation time series 
  (seaonsal and deseasonalised). Plots are labelled with the AR1 of the 
  vegetation time series, and the maximum correlation between the Vegetation
  and precipitation time series.
- Auto-correlation plots for vegetation and precipitation time series
  (seaonsal and deseasonalised).
- Vegetation and precipitation cross-correlation scatterplot matrices.
- STL decomposition plots.
- Resiliance analysis:
     - `ewstools` resiliance plots showing AR1, standard deviation, 
       skewness, and kurtosis using a moving window.
     - Smoothing filter size and moving window size Kendall tau 
       sensitivity plots.
     - Signficance test.

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

## `pyveg` flow

The digram below represents the high level flow of the `pyveg` package.

![`pyveg` program flow.](paper/pveg_flow.png)


# Licence

This project is licensed under the terms of the MIT software license.