#!/usr/bin/env python

"""
Scripts to process the output of the GEE images and json files with network centrality.

The json outputs are turned into a dataframe and the values of a particular metric a plotted
as a function of time.

Finally a GIF file is produced with all of the network metric images, as well as the original 10km x 10km dowloaded images.
"""

import argparse

from pyveg.src.process_network_metrics import (
    process_json_metrics_to_dataframe,
    create_network_figures
)

from pyveg.src.image_utils import (
    create_gif_from_images
)


def main():
    """
    use command line arguments to choose images.
    """
    parser = argparse.ArgumentParser(description="process json files with network centrality measures from from GEE images")
    parser.add_argument("--input_dir",help="input directory where all GEE download files are found",
                        default=".")
    parser.add_argument("--output_dir",help="output directory",
                        default=".")
    parser.add_argument("--metric_name",help="name of metric used for the analysis",
                        default="offset50")
    parser.add_argument("--output_name",help="name of output filename, not including file extension",
                      default="file")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_name = args.output_name
    metric_name = args.metric_name

    metrics_df = process_json_metrics_to_dataframe(input_dir)

    create_network_figures(metrics_df, metric_name, output_dir, output_name)

    create_gif_from_images(output_dir, output_dir+output_name)

    # create gif evolution for the 10km images for references
    create_gif_from_images(input_dir, output_dir+output_name+"_Images10Km","10kmLargeImage")


    print("Done")




if __name__ == "__main__":
    main()
