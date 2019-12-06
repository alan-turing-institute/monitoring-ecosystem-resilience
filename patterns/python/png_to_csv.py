#!/usr/bin/env python
"""
Quick script to convert a black+white png file into a csv file containing
0 (black) and 255 (white) by default.
Can be configured to have different max value via the --maxval argument,
and to swap black and white via the --invert argument.
"""
import os
import argparse
from PIL import Image

def convert(inputfile, outputfile, maxval, invert=False):
    img = Image.open(inputfile)
    pix = img.load()
    outfile = open(outputfile,"w")
    for ix in range(img.size[0]):
        line = ""
        for iy in range(img.size[1]):
            pixval = sum(pix[ix,iy])
            if (pixval == 0 and not invert) or \
               (pixval > 0 and invert):
                line += "0,"
            else:
                line += "{},".format(maxval)
        line = line[:-1]+"\n"
        outfile.write(line)
    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert black+white png to csv")
    parser.add_argument("--input",help="input png file", required=True)
    parser.add_argument("--output",help="output csv file")
    parser.add_argument("--maxval",help="value for non-zero pixels",default=255)
    parser.add_argument("--invert",help="swap black and white", action="store_true")
    args = parser.parse_args()
    input_filename = args.input
    if args.output:
        output_filename = args.output
    else:
        output_path = os.path.split(input_filename)[0]
        output_filename_parts = os.path.basename(input_filename).split(".")
        output_filename = ""
        for part in output_filename_parts:
            output_filename += part + "."
        output_filename = output_filename[:-1] +".csv"
        output_filename = os.path.join(output_path, output_filename)
    maxval = args.maxval
    invert = args.invert if args.invert else False
    convert(input_filename, output_filename, maxval, invert)
