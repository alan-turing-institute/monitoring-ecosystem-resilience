#!/usr/bin/env python

"""
Translation of Matlab code to model patterned vegetation in semi-arid landscapes.
"""

import argparse

from pyveg.src.pattern_generation import PatternGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate vegetation patterns")
    parser.add_argument("--rainfall", help="rainfall in mm",
                        type=float, default=1.0)
    parser.add_argument(
        "--input_config", help="input config JSON filename", type=str)
    parser.add_argument(
        "--input_csv", help="starting pattern CSV filename", type=str)
    parser.add_argument(
        "--steps", help="number of time steps to run", type=int, default=1000
    )
    parser.add_argument(
        "--transpose",
        help="rotate image (useful for comparing to matlab",
        action="store_true",
    )
    parser.add_argument(
        "--make_binary", help="threshold the plant_biomass member", action="store_true"
    )
    parser.add_argument("--output_png", help="output png filename", type=str)
    parser.add_argument("--output_csv", help="output csv filename", type=str)
    parser.add_argument("--output_matlab", help="output .m filename", type=str)
    parser.add_argument(
        "--plot_result", help="display the evolved pattern", action="store_true"
    )
    args = parser.parse_args()

    print("-" * 45)
    print("Starting pattern generator...")
    print("-" * 45)

    pg = PatternGenerator()
    pg.set_rainfall(args.rainfall)

    if args.input_config:
        pg.load_config(args.input_config)

    pg.initial_conditions()

    if args.input_csv:
        pg.set_starting_pattern_from_file(args.input_csv)
    else:
        pg.set_random_starting_pattern()

    pg.evolve_pattern(steps=args.steps)

    if args.make_binary:
        pg.plant_biomass = pg.make_binary()
    if args.transpose:
        pg.plant_biomass = pg.plant_biomass.transpose()
    if args.plot_result:
        pg.plot_image()

    if args.output_csv:
        pg.save_as_csv(args.output_csv)
    if args.output_png:
        pg.save_as_png(args.output_png)
    if args.output_matlab:
        pg.save_as_matlab(args.output_matlab)

    print("Finished generating patterns!\n")


if __name__ == "__main__":
    main()
