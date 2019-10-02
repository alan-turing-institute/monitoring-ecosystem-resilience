"""
Modify, and slice up tif and png images using Python Image Library
Needs a relatively recent version of pillow (fork of PIL):
```
pip install --upgrade pillow
```
"""

import sys
from PIL import Image
import sys

def combine_tif(input_filebase, bands=["B4","B3","B2"]):
    """
    Read tif files in "I" mode - one per specified band, and rescale and combine
    pixel values to r,g,b values betweek 0 and 255 in a combined output image.
    Currently assumes that we have three bands.  Need to figure out how to
    deal with more or fewer...
    """
    if len(bands) >= 3:
        band_dict = {"r": {"band": bands[0],
                           "min_val": sys.maxsize,
                           "max_val": -1*sys.maxsize,
                           "pix_vals": []},
                     "g": {"band": bands[1],
                           "min_val": sys.maxsize,
                           "max_val": -1*sys.maxsize,
                           "pix_vals": []},
                     "b": {"band": bands[2],
                           "min_val": sys.maxsize,
                           "max_val": -1*sys.maxsize,
                           "pix_vals": []}
        }
    for col in band_dict.keys():
        im = Image.open(input_filebase+"."+band_dict[col]["band"]+".tif")
        pix = im.load()
        ## find the minimum and maximum pixel values in the original scale
        print("Found image of size {}".format(im.size))
        for ix in range(im.size[0]):
            for iy in range(im.size[1]):
                if pix[ix,iy]> band_dict[col]["max_val"]:
                    band_dict[col]["max_val"]= pix[ix,iy]
                elif pix[ix,iy] < band_dict[col]["min_val"]:
                    band_dict[col]["min_val"] = pix[ix,iy]
        band_dict[col]["pix_vals"] = pix
    # Take the overall max of the three bands to be the value to scale down with.
    print("Max values {} {} {}".format(band_dict["r"]["max_val"],
                                       band_dict["g"]["max_val"],
                                       band_dict["b"]["max_val"]))

    overall_max = max((band_dict[col]["max_val"] for col in ["r","g","b"]))

    # create a new image where we will fill RGB pixel values from 0 to 255
    get_pix_val = lambda ix, iy, col: \
        max(0, int(band_dict[col]["pix_vals"][ix,iy] * 255/ \
#                   band_dict[col]["max_val"]
                   (overall_max+1)
        ))
    new_img = Image.new("RGB", im.size)
    for ix in range(im.size[0]):
        for iy in range(im.size[1]):
            new_img.putpixel((ix,iy), tuple(get_pix_val(ix,iy,col) \
                                            for col in ["r","g","b"]))
    return new_img


def crop_image(input_image, n_parts_x, n_parts_y=None):
    """
    Divide an image into smaller sub-images.
    """
    ## if n_parts_y not specified, assume we want equal x,y
    if not n_parts_y:
        n_parts_y = n_parts_x

    xsize, ysize = input_image.size
    x_sub = int(xsize / n_parts_x)
    y_sub = int(ysize / n_parts_y)


    sub_images = []
    for ix in range(n_parts_x):
        for iy in range(n_parts_y):
            box = (ix*x_sub, iy*y_sub, (ix+1)*x_sub, (iy+1)*y_sub)
            region = input_image.crop(box)
            sub_images.append(region)

    return sub_images


def convert_to_bw(input_image, threshold):
    """
    Given an RGB input, apply a threshold to each pixel.
    If pix(r,g,b)>threshold, set to 255,255,255, if <threshold, set to 0,0,0
    """
    pix = input_image.load()
    new_img = Image.new("RGB", input_image.size)
    for ix in range(input_image.size[0]):
        for iy in range(input_image.size[1]):
            p = pix[ix,iy]
            total = 0
            for col in p:
                total += col
            if total > threshold:
                new_img.putpixel((ix,iy), (255,255,255))
            else:
                new_img.putpixel((ix,iy), (0,0,0))
    return new_img
