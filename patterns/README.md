# Modelling vegetation patterns in arid landscapes

The code in this repository is a python translation (and R translation-in-progress) of some Matlab code for simulating and analysing vegetation patterns.

### Python setup

In order to have a clean working environment it is useful to have [anaconda](https://www.anaconda.com/distribution/) installed.
Create an anaconda environment using the environment.yml file in this directory. Concretely:
```
    conda update conda
    conda env create -f environment.yml
    source activate patterns
 ```
If you don't have anaconda, you can also install the requirements by doing
```
pip install -r requirements.txt
```

### R setup

(Note that we currently only have an R implementation of the pattern generation, not the network modelling yet).

Dependencies:
```
dplyr
ggplot2
gganimate
jsonlite
```
These can be installed from CRAN via ```install.packages(<package_name>)```.


## Pattern simulation

### Pattern simulation in Python

The ```generate_patterns.py``` functionality originates from some Matlab code by Stefan Dekker, Willem Bouten, Maarten Boerlijst and Max Rietkerk (included in the "matlab" directory), implementing the scheme described in:

Rietkerk et al. 2002. Self-organization of vegetation in arid ecosystems. The American Naturalist 160(4): 524-530.

To run this simulation in Python, change to the `python` subdirectory and run
```
python generate_patterns.py --help
```
to see the options.  The most useful option is the `--rainfall` parameter which sets a parameter (the rainfall in mm) of the simulation - values between 1.2 and 1.5 seem to give rise to a good range of patterns.
Other optional parameters for `generate_patterns.py` allow the generated image to be output as a csv file or a png image.  The `--transpose` option rotates the image 90 degrees (this was useful for comparing the Python and Matlab network-modelling code).
Other parameters for running the simulation are in the file `config.py`, you are free to change them.

### Pattern simulation in R


To run:
From *R* or *RStudio*
```
source('patterns.R')
plot
```
Configuration parameters are loaded from the file ```config.json```.


## Network modelling in Python

This approach is based on the Matlab file `mao_pollen_EC.m` written by Luke Mander to look at connectedness of
pixels on a binary image, using "Subgraph Centrality" as described in:

Mander et.al. "A morphometric analysis of vegetation patterns in dryland ecosystems",
R. Soc. open sci. (2017)
https://royalsocietypublishing.org/doi/10.1098/rsos.160443

Mander et.al. "Classification of grass pollen through the quantitative
analysis of surface ornamentation and texture", Proc R Soc B 280: 20131905.
https://royalsocietypublishing.org/doi/pdf/10.1098/rspb.2013.1905

Estrada et.al. "Subgraph Centrality in Complex Networks"
https://arxiv.org/pdf/cond-mat/0504730.pdf

To run the python version, from the `python` subdirectory, do
```
python subgraph_centrality.py --help
```
to see the available options.

The options are:
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
python -i subgraph_centrality.py --input_txt ../binary_image.txt --do_EC
>>> sc_images[50].show() # plot the image with the top 50% of pixels (ordered by subgraph centrality) highlighted.
>>> plt.plot(list(sel_pixels.keys()), feature_vec, "bo") # plot the feature vector vs pixel rank
>>> plt.show()
```

## Putting it all together

To run both the pattern generation and the subgraph-centrality modelling together, run the python script
```
python plot_feature_vectors.py --help
```
to see the available options.
These are:
* `--rainfall_vals` comma-separated list of floats representing the rainfall values to use in the simulation.
* `--do_EC` calculate the Euler Characteristic as the values in the feature vector.  Currently required, as the other option is not yet fully debugged.

Again, it is recommended to use the `-i` flag to get an interactive python session if you would like to modify the cosmetics of the output plot.

Example:
```
python -i plot_feature_vectors.py --rainfall_vals 1.2,1.3,1.4 --do_EC
```
