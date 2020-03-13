#!/usr/bin/env python

"""
Use some of the functionality in src/image_utils.py as command-line script.
"""


import argparse

from pyveg.src.image_utils import *


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
