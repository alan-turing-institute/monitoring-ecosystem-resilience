#!/usr/bin/env python

"""
Try to reproduce plots from Mander et. al. "A morphometric analysis of vegetatoin patterns in dryland ecosystems"
"""

import os
import matplotlib.pyplot as plt
import argparse

from generate_patterns import generate_pattern
from subgraph_centrality import subgraph_centrality, generate_sc_images

LABELS=['bo','ro','go', 'b^', 'r^', 'g^']


def feature_vec_plot(rainfall, do_EC=True):
    """
    Generate a pattern and then calculate the feature vector
    """
    image = generate_pattern(rainfall)
    fv, sc = subgraph_centrality(image, do_EC, threshold=255)
    # ignore the first element
    xvals = list(sc.keys())[1:]
    yvals = list(fv[1:])
    return xvals, yvals, image


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
        xvals[label], yvals[label], images[label] = feature_vec_plot(float(rain), do_EC)

    for i, rain in enumerate(xvals.keys()):
        plt.plot(xvals[rain], yvals[rain], LABELS[i])


    plt.xlabel("pixel rank (%)")
    plt.ylabel("Euler characteristic")
