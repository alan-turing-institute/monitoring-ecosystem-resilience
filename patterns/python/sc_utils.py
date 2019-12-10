"""
A set of simple utility functions (mainly reading and writing
files and converting images to arrays and vice versa
"""

### First set of functions read either a csv file or png into
### a 2D numpy array of pixel values.

import numpy as np
from PIL import Image

def read_image_file(input_filename):
    """
    Read an image and convert to a 2D numpy array, with values
    0 for background pixels and 255 for signal.
    Assume that the input image has only two colours, and take
    the one with higher sum(r,g,b) to be "signal".
    """
    im = Image.open(input_filename)
    x_size, y_size = im.size
    pix = im.load()
    sig_col = None
    bkg_col = None
    max_sum_rgb = 0
    # loop through all the pixels and find the colour with the highest sum(r,g,b)
    for ix in range(x_size):
        for iy in range(y_size):
            col = pix[ix,iy]
            if sum(col) > max_sum_rgb:
                max_rgb = sum(col)
                if sig_col:
                    bkg_col = sig_col
                sig_col = col
    # ok, we now know what sig_col is - loop through pixels again and set any that
    # match this colour to 255.
    rows = []
    for iy in range(y_size):
        row = np.zeros(x_size)
        for ix in range(x_size):
            if pix[ix,iy] == sig_col:
                row[ix] = 255
        rows.append(row)
    return np.array(rows)


def read_text_file(input_filename):
    """
    Read a csv-like representation of an image, where each row (representing a row
    of pixels in the image) is a comma-separated list of pixel values 0 (for black)
    or 255 (for white).
    """
    input_image = np.loadtxt(input_filename, dtype='i', delimiter=',')
    return input_image


def crop_image(input_image, x_range, y_range):
    """
    return a new image from specified pixel range of input image
    """
    return input_image[x_range[0]:x_range[1], y_range[0]:y_range[1]]



## Write results of the SC analysis to a CSV

def write_csv(feature_vec, output_filename):
    """
    Write the feature vector to a 1-line csv
    """
    output_string = ""
    for feat in feature_vec:
        output_string += str(feat)+","
    output_string = output_string[:-1]+"\n"
    with open(output_filename,"w") as outfile:
        outfile.write(output_string)
    return True


def write_dict_to_csv(metrics_dict, output_filename):
    with open(output_filename, 'w') as f:
        for key in metrics_dict.keys():
            f.write("%s,%s\n" % (key, metrics_dict[key]))
    return True


## A set of functions to help visualize the subgraphs, similar
## to the plots in Fig.3 of the Mander et.al. paper.

def image_from_array(input_array, output_size=None, sel_val=200):
    """
    Convert a 2D numpy array of values into
    an image where each pixel has r,g,b set to
    the corresponding value in the array.
    If an output size is specified, rescale to this size.
    """
    size_x, size_y = input_array.shape
    new_img = Image.new("RGB", (size_x,size_y))
    # count the number of distinct values in the array
    for ix in range(size_x):
        for iy in range(size_y):
            val = int(input_array[ix,iy])

            if val == sel_val:
                new_img.putpixel((ix,iy),(0,val,val))
            else:
                new_img.putpixel((ix,iy),(val,val,val))
    if output_size:
        new_img = new_img.resize((output_size, output_size), Image.ANTIALIAS)
    return new_img


def fill_sc_pixels(sel_pixels, orig_image, val=200):
    """
    Given an original 2D array where all the elements are 0 (background)
    or 255 (signal), fill in a selected subset of signal pixels as 123 (grey).
    """
    new_image = np.copy(orig_image)
    for pix in sel_pixels:
        new_image[pix] = 200
    return new_image


def generate_sc_images(sel_pixels, orig_image, val=200):
    """
    Return a dict of images with the selected subsets of signal
    pixels filled in in cyan.
    """
    image_dict = {}
    for k,v in sel_pixels.items():
        new_image_array = fill_sc_pixels(v, orig_image, val)
        new_image = image_from_array(new_image_array,400)
        image_dict[k] = new_image
    return image_dict


def save_sc_images(image_dict, file_prefix):
    """
    Saves images from dictionary.
    """
    for key in image_dict:
        im = image_dict[key]
        im.save(file_prefix + "_"+ str(key), "JPEG")
