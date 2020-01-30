#!/usr/bin/env python

"""
Translation of Matlab code to model patterned vegetation in semi-arid landscapes.
"""

import argparse
from pyveg.src.pattern_generation import PatternGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate vegetation patterns")
    parser.add_argument("--rainfall", help="rainfall in mm",type=float, default=1.4)
    parser.add_argument("--input_config", help="input config JSON filename",type=str)
    parser.add_argument("--input_csv", help="starting pattern CSV filename",type=str)
    parser.add_argument("--output_png", help="output png filename",type=str)
    parser.add_argument("--output_csv", help="output csv filename",type=str)
    parser.add_argument("--steps", help="number of time steps to run",type=int, default=10000)
    parser.add_argument("--transpose", help="rotate image (useful for comparing to matlab",action="store_true")
    args = parser.parse_args()

    pg = PatternGenerator()
    pg.set_rainfall(args.rainfall)
    if args.input_config:
        pg.set_config(args.input_config)
    pg.initial_conditions()
    if args.input_csv:
        pg.set_starting_pattern_from_file(args.input_csv)
    else:
        pg.set_random_starting_pattern()

    pg.evolve_pattern(steps=args.steps)

    binary_pattern = pg.make_binary()
    if args.transpose:
        binary_pattern = binary_pattern.transpose()
    if args.output_csv:
        save_as_csv(binary_pattern, args.output_csv)
    if args.output_png:
        save_as_png(binary_pattern, args.output_png)
    plot_image(binary_pattern)


if __name__ == "__main__":
    main()
