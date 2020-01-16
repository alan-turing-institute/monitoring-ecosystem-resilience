#!/usr/bin/env python

"""
Translation of Matlab code to model patterned vegetation in semi-arid landscapes.
"""

import argparse
from pyveg.src.pattern_generation import (
    generate_pattern,
    plot_image,
    save_as_csv,
    save_as_png
)

def main():
    parser = argparse.ArgumentParser(description="Generate vegetation patterns")
    parser.add_argument("--rainfall", help="rainfall in mm",type=float, default=1.4)
    parser.add_argument("--output_png", help="output png filename",type=str)
    parser.add_argument("--output_csv", help="output csv filename",type=str)
    parser.add_argument("--transpose", help="rotate image (useful for comparing to matlab",action="store_true")
    args = parser.parse_args()

    binary_pattern = generate_pattern(args.rainfall)
    if args.transpose:
        binary_pattern = binary_pattern.transpose()
    if args.output_csv:
        save_as_csv(binary_pattern, args.output_csv)
    if args.output_png:
        save_as_png(binary_pattern, args.output_png)
    plot_image(binary_pattern)


if __name__ == "__main__":
    main()
