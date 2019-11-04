#!/usr/bin/env python

"""
Try to reproduce plots from Mander et. al. "A morphometric analysis of vegetatoin patterns in dryland ecosystems"
"""

import os
import matplotlib.pyplot as plt
import argparse

from generate_patterns import generate_pattern
from subgraph_centrality import (
    subgraph_centrality,
    generate_sc_images,
    image_from_array
)

LABELS=['bo','ro','go', 'b^', 'r^', 'g^']


def generate_feature_vec_plot(rainfall, do_EC=True):
    """
    Generate a pattern and then calculate the feature vector
    """
    image = generate_pattern(rainfall)
    fv, sc = subgraph_centrality(image, do_EC, threshold=255)
    # ignore the first element
    xvals = list(sc.keys())[1:]
    yvals = list(fv[1:])
    images = generate_sc_images(sc, image)
    return xvals, yvals, images


def display_plots(xvals, yvals):
    """
    Show the plots on the same canvas
    """
    for i, rain in enumerate(xvals.keys()):
        plt.plot(xvals[rain], yvals[rain], LABELS[i])

    plt.xlabel("pixel rank (%)")
    plt.ylabel("Euler characteristic")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot feature vector")
    parser.add_argument("--rainfall_vals",help="comma-separated list of rainfall vals")
    parser.add_argument("--do_EC",help="calc Euler characteristic",action="store_true")
    args = parser.parse_args()
    rainfall_vals = args.rainfall_vals.split(",")
    do_EC = True if args.do_EC else False
    images = {}
    xvals = {}
    yvals = {}
    for rain in rainfall_vals:
        label = "{}mm".format(rain)
        xvals[label], yvals[label], images[label] = generate_feature_vec_plot(float(rain),
                                                                              do_EC)
