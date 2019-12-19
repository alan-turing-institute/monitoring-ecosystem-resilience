#!/usr/bin/env python

"""
Try to reproduce plots from Mander et. al. "A morphometric analysis of vegetatoin patterns in dryland ecosystems"

Example usage:
```
python -i plot_feature_vectors.py --rainfall_vals 1.1,1.2,1.3,1.4,1.5,1.55
```
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


def generate_feature_vec_plot(rainfall):
    """
    Generate a pattern and then calculate the feature vector
    """
    image = generate_pattern(rainfall)
    fv, sc = subgraph_centrality(image, threshold=255)
    # ignore the first element
    xvals = list(sc.keys())[1:]
    yvals = list(fv[1:])
    images = generate_sc_images(sc, image)
    return xvals, yvals, images


def display_plots(xvals, yvals):
    """
    Show the plots on the same canvas
    """
    for i, rain in enumerate(yvals.keys()):
        plt.plot(xvals, yvals[rain], LABELS[i], label=rain)
    plt.legend()
    plt.xlabel("pixel rank (%)")
    plt.ylabel("Euler characteristic")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot feature vector")
    parser.add_argument("--rainfall_vals",help="comma-separated list of rainfall vals")
    args = parser.parse_args()
    rainfall_vals = args.rainfall_vals.split(",")
    images = {}
    xvals = []
    yvals = {}
    for rain in rainfall_vals:
        label = "{}mm".format(rain)
        xvals, yvals[label], images[label] = generate_feature_vec_plot(float(rain))
    display_plots(xvals, yvals)
