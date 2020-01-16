"""
Modify, and slice up tif and png images using Python Image Library
Needs a relatively recent version of pillow (fork of PIL):
```
pip install --upgrade pillow
```
"""

import os
import sys
import argparse

from PIL import Image
import matplotlib.pyplot as plt


def save_image(image, output_dir, output_filename):
    """
    Given a PIL.Image (list of pixel values), save
    to requested filename - note that the file extension
    will determine the output file type, can be .png, .tif,
    probably others...
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    image.save(output_path)
    print("Saved image {}".format(output_path))


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



def image_file_to_array(input_filename):
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


def invert_binary_image(image):
    """
    Swap (255,255,255) with (0,0,0) for all pixels
    """
    new_img = Image.new("RGB", image.size)
    pix = image.load()
    for ix in range(image.size[0]):
        for iy in range(image.size[1]):
            if sum(pix[ix,iy]) == 0:
                new_img.putpixel((ix,iy), (255,255,255))
            else:
                new_img.putpixel((ix,iy), (0,0,0))
    return new_img


def scale_tif(input_filebase, band):
    """
    Given only a single band, scale to range 0,255 and apply this
    value to all of r,g,b
    """
    max_val = -1*sys.maxsize
    min_val =  sys.maxsize
    # load the single band file and extract pixel data
    im = Image.open(input_filebase+"."+band+".tif")
    pix = im.load()
    ## find the minimum and maximum pixel values in the original scale
    print("Found image of size {}".format(im.size))
    for ix in range(im.size[0]):
        for iy in range(im.size[1]):
            if pix[ix,iy] > max_val:
                max_val = pix[ix,iy]
            elif pix[ix,iy] < min_val:
                min_val = pix[ix,iy]

    # create a new image where we will fill RGB pixel values from 0 to 255
    get_pix_val = lambda ix, iy: \
        max(0, int((pix[ix,iy]-min_val) * 255/ \
                (max_val - min_val))
        )
    new_img = Image.new("RGB", im.size)
    for ix in range(im.size[0]):
        for iy in range(im.size[1]):
            new_img.putpixel((ix,iy), tuple(get_pix_val(ix,iy) \
                                            for col in ["r","g","b"]))
    return new_img


def plot_band_values(input_filebase, bands=["B4","B3","B2"]):
    """
    Plot histograms of the values in the chosen bands of the input image
    """
    num_subplots = len(bands)
    for i, band in enumerate(bands):
        im = Image.open(input_filebase+"."+band+".tif")
        pix = im.load()
        vals = []
        for ix in range(im.size[0]):
            for iy in range(im.size[1]):
                vals.append(pix[ix,iy])
        plt.subplot(1,num_subplots, i+1)
        plt.hist(vals)
    plt.show()


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


def crop_image_npix(input_image, n_pix_x, n_pix_y=None,
                    region_size=None, coords=None):
    """
    Divide an image into smaller sub-images with fixed pixel size.
    If region_size and coordinates are provided, we want to return the
    coordinates of the sub-images along with the sub-images themselves.
    """
    ## if n_pix_y not specified, assume we want equal x,y
    if not n_pix_y:
        n_pix_y = n_pix_x

    xsize, ysize = input_image.size
    x_parts = int(xsize // n_pix_x)
    y_parts = int(ysize // n_pix_y)

    # if we are given coords, calculate coords for all sub-regions
    sub_image_coords = []
    if coords and region_size:
        left_start = coords[0] - region_size/2
        bottom_start = coords[1] - region_size/2
        sub_image_size_x = region_size / x_parts
        sub_image_size_y = region_size / y_parts
        for ix in range(x_parts):
            for iy in range(y_parts):
                sub_image_coords.append(
                    (left_start + sub_image_size_x/2 + (ix*sub_image_size_x),
                     bottom_start + sub_image_size_y/2 + (iy*sub_image_size_y))
                )

    # now do the actual cropping
    sub_images = []
    for ix in range(x_parts):
        for iy in range(y_parts):
            box = (ix*n_pix_x, iy*n_pix_y, (ix+1)*n_pix_x, (iy+1)*n_pix_y)
            region = input_image.crop(box)
            # depending on whether we have been given coordinates,
            # return a list of images, or a list of (image,coords) tuples.
            if sub_image_coords:
                sub_images.append((region, sub_image_coords[ix*x_parts+iy]))
            else:
                sub_images.append(region)

    return sub_images



def crop_image_nparts(input_image, n_parts_x, n_parts_y=None):
    """
    Divide an image into n_parts_x*n_parts_y equal smaller sub-images.
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


def convert_to_bw(input_image, threshold, invert=False):
    """
    Given an RGB input, apply a threshold to each pixel.
    If pix(r,g,b)>threshold, set to 255,255,255, if <threshold, set to 0,0,0
    """
    pix = input_image.load()
    new_img = Image.new("RGB", input_image.size)
    for ix in range(input_image.size[0]):
        for iy in range(input_image.size[1]):
            p = pix[ix,iy]
            try:
                total = 0
                for col in p:
                    total += col
            except:
                total = p
            if (invert and (total > threshold)) or \
               ((not invert) and (total < threshold)):
                new_img.putpixel((ix,iy), (255,255,255))
            else:
                new_img.putpixel((ix,iy), (0,0,0))
    return new_img



def crop_and_convert_to_bw(input_filename, output_dir, threshold=470, num_x=50, num_y=50):
    """
    Open an image file, convert to monochrome, and crop into sub-images.
    """
    orig_image = Image.open(input_filename)
    bw_image = convert_to_bw(orig_image, threshold)
    sub_images = crop_image_npix(bw_image, num_x, num_y)
    ## strip the file extension from the input_filename
    filename_elements = os.path.basename(input_filename).split(".")
    file_ext = filename_elements[-1]
    new_filename_base = ""
    for el in filename_elements[:-1]:
        new_filename_base+= el

    for i, sub_image in enumerate(sub_images):
        new_filename = "{}_{}.{}".format(new_filename_base,
                                         i,
                                         file_ext)
        save_image(sub_image, output_dir, new_filename)


def crop_and_convert_all(input_dir, output_dir, threshold=470, num_x=50, num_y=50):
    """
    Loop through a whole directory and crop and convert to black+white all
    files within it.
    """
    for filename in os.listdir(input_dir):
        if not (filename.endswith("tif") or filename.endswith("png")):
            continue
        print("Processing {}".format(filename))
        input_filename = os.path.join(input_dir, filename)
        crop_and_convert_to_bw(input_filename, output_dir,
                               threshold, num_x, num_y)


def main():
    """
    use command line arguments to specify input and output directories,
    and parameters for doing the cropping and conversion.
    """
    parser = argparse.ArgumentParser(description="crop and convert images")
    parser.add_argument("--input_dir",help="full path to directory containing input images", required=True)
    parser.add_argument("--output_dir",help="directory to put output images", required=True)

    parser.add_argument("--threshold",help="sum(r,g,b) threshold above which we colour the pixel white, or below black",
                        default=470, type=int)
    parser.add_argument("--num_pix_x",help="Size in pixels of cropped image along x-axis",
                        default=50, type=int)
    parser.add_argument("--num_pix_y",help="Size in pixels of cropped image along y-axis",
                        default=50, type=int)
    args = parser.parse_args()
    ## now call the crop_and_convert function
    crop_and_convert_all(args.input_dir, args.output_dir, args.threshold,
                         args.num_pix_x, args.num_pix_y)


##########################

if __name__ == "__main__":
    main()
