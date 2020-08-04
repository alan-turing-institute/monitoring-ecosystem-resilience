#!/usr/bin/env python

"""
script to run the Euler Characteristic calculation on a given input
image.
The code that does the actual calculations is in
src/subgraph_centrality.py
"""

import os
import argparse

from pyveg.src.subgraph_centrality import (
    subgraph_centrality,
    feature_vector_metrics,
    generate_sc_images,
    text_file_to_array,
    image_file_to_array,
)


def main():
    parser = argparse.ArgumentParser(
        description="Look at subgraph centrality of signal pixels in an image"
    )
    parser.add_argument("--input_txt", help="input image as a csv file")
    parser.add_argument("--input_img", help="input image as an image file")
    parser.add_argument(
        "--use_diagonal_neighbours",
        help="use 8-neighbours rather than 4-neighbours",
        action="store_true",
    )
    parser.add_argument(
        "--num_quantiles",
        help="number of elements of feature vector",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--sig_threshold", help="threshold for signal pixel", type=int, default=255
    )
    parser.add_argument(
        "--upper_threshold",
        help="threshold for signal pixel is an upper limit",
        action="store_true",
    )
    parser.add_argument(
        "--output_csv",
        help="filename for output csv of feature vector",
        default="feature_vector.csv",
    )
    parser.add_argument("--output_img", help="filename for output images")

    args = parser.parse_args()
    image_array = None
    if args.input_txt:
        image_array = text_file_to_array(args.input_txt)
    elif args.input_img:
        image_array = image_file_to_array(args.input_img)
    else:
        raise RuntimeError("Need to specify input_txt or input_img")
    use_diagonal_neighbours = True if args.use_diagonal_neighbours else False
    num_quantiles = args.num_quantiles
    threshold = args.sig_threshold
    is_lower_limit = True if not args.upper_threshold else False
    output_csv = args.output_csv
    # call the subgraph_centrality function to calculate everything
    feature_vec, sel_pixels = subgraph_centrality(
        image_array,
        use_diagonal_neighbours,
        num_quantiles,
        threshold,
        is_lower_limit,
        output_csv,
    )
    # get the images showing the selected sub-regions
    sc_images = generate_sc_images(sel_pixels, image_array)

    feature_vec_metrics = feature_vector_metrics(feature_vec, output_csv)

    print(feature_vec_metrics)

    if args.output_img:
        save_sc_images(sc_images, args.output_img)


if __name__ == "__main__":
    main()
