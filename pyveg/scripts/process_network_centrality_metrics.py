#!/usr/bin/env python

"""
Scripts to process the output of the GEE images and json files with network centrality measures.

The json outputs are turned into a dataframe and the values of a particular metric a plotted
as a function of time.

Finally a GIF file is produced with all of the network metric images, as well as the original 10km x 10km dowloaded images.
"""

import argparse

from pyveg.src.data_analysis_utils import (
    read_json_to_dataframe,
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
    parser.add_argument("--output_dir",help="directory to save outputs",
                        default=".")
    parser.add_argument("--metric_name",help="name of metric used for the analysis (eg. offset50, slope, mean)",
                        default="offset50")
    parser.add_argument("--output_name",help="name of output filename, not including file extension",
                      default="file")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_name = args.output_name
    metric_name = args.metric_name

    # read all json files in the directory and produce a dataframe
    metrics_df = read_json_to_dataframe(input_dir)

    # from the dataframe, produce network metric figure for each avalaible date
    create_network_figures(metrics_df, metric= metric_name, output_dir = output_dir, output_name= output_name)

    # get all figures into a gif file
    create_gif_from_images(output_dir, output_name)


    # create gif evolution for the 10km images for reference
    create_gif_from_images(input_dir, output_name+"_Images10Km_ndvi","10kmLargeImage_ndvi_")

    create_gif_from_images(input_dir, output_name+"_Images10Km_colour_","10kmLargeImage_colour")

    create_gif_from_images(input_dir, output_name+"_Images10Km_ndvibw_","10kmLargeImage_ndvibw")

    metrics_df.to_csv(output_dir+"metrics_df.csv")

    print("Done")



if __name__ == "__main__":
    main()
