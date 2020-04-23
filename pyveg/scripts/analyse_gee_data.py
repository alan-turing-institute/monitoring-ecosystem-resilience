#!/usr/bin/env python

"""
Scripts to process the output of the GEE images and json files with network centrality measures.

The json outputs are turned into a dataframe and the values of a particular metric a plotted
as a function of time.

Finally a GIF file is produced with all of the network metric images, as well as the original 10km x 10km dowloaded images.
"""

import argparse
import os

import pandas as pd

from pyveg.src.analysis_preprocessing import preprocess_data

from pyveg.src.data_analysis_utils import (
    create_lat_long_metric_figures,
    convert_to_geopandas,
    coarse_dataframe,
    remove_seasonality_combined,
    remove_seasonality_all_sub_images,
)
from pyveg.src.plotting import (
    do_stl_decomposition,
    plot_smoothed_time_series,
    plot_autocorrelation_function,
    plot_feature_vector,
    plot_cross_correlations
)


def analyse_gee_data(input_dir, spatial):

    """
    Run analysis on dowloaded gee data

    Parameters
    ----------
    input_dir : string
        Path to directory with downloaded dada
    do_spatial_plot: bool
        Option to run spatial analysis and do plots
    do_time_series_plot: bool
        Option to run time-series analysis and do plots

    """

    ts_filename, dfs = preprocess_data(input_dir)
    ts_df = pd.read_csv(ts_filename)

    # put output plots in the results dir
    output_dir = os.path.join(input_dir, 'analysis')

    # spatial analysis and plotting
    # ------------------------------------------------
    if spatial:

        # from the dataframe, produce network metric figure for each avalaible date
        print('\nCreating spatial plots...')

        # create new subdir for time series analysis
        spatial_subdir = os.path.join(output_dir, 'spatial')
        if not os.path.exists(spatial_subdir):
            os.makedirs(spatial_subdir, exist_ok=True)

        for collection_name, df in dfs.items():
            if collection_name == 'COPERNICUS/S2' or 'LANDSAT' in collection_name:
                data_df_geo = convert_to_geopandas(df.copy())
                data_df_geo_coarse = coarse_dataframe(data_df_geo.copy(), 2)
                create_lat_long_metric_figures(data_df_geo_coarse, 'offset50', spatial_subdir)
    # ------------------------------------------------

    # ------------------------------------------------
    # main analysis and plotting sequence
    # ------------------------------------------------
    # create new subdir for time series analysis
    # tsa_subdir = os.path.join(output_dir, 'time-series') # if we start to have more and more results
    tsa_subdir = output_dir

    if not os.path.exists(tsa_subdir):
        os.makedirs(tsa_subdir, exist_ok=True)

    # feature vectors
    # ------------------------------------------------
    # plot the feature vectors
    plot_feature_vector(dfs, tsa_subdir) # TODO: read vector vectors from csv, and plot extreme feature vectors (issue #186)

    # auto- and cross-correlations
    # --------------------------------------------------
    # make autocorrelation plots
    plot_autocorrelation_function(ts_df, tsa_subdir)

    # make cross correlation scatterplot matrix plots
    plot_cross_correlations(ts_df, tsa_subdir)
    # --------------------------------------------------

    # time series
    # ------------------------------------------------
    # make a smoothed time series plot
    plot_smoothed_time_series(ts_df, tsa_subdir)




    """do_stl_decomposition(time_series_dfs, 12, tsa_subdir)

    # --------------------------------------------------
    #   remove seasonality in a time series
    time_series_uns_dfs = remove_seasonality_all_sub_images(smooth_veg_data(dfs.copy(), n=4), 12, "M")

    smoothed_time_series_uns_dfs = make_time_series(time_series_uns_dfs.copy())  # increase smoothing with n>5

    # make a smoothed time series plot
    plot_smoothed_time_series(smoothed_time_series_uns_dfs, tsa_subdir, '-no-seasonality')

    # make autocorrelation plots
    plot_autocorrelation_function(smoothed_time_series_uns_dfs, tsa_subdir, '-no-seasonality')

    # write csv for easy external analysis
    write_slimmed_csv(smoothed_time_series_uns_dfs, tsa_subdir, '-no-seasonality')

    # ------------------------------------------------
    #   remove seasonality in the summary time series
    time_series_uns_summary_dfs = remove_seasonality_combined(smoothed_time_series_dfs.copy(), 12, "M")

    # make a smoothed time series plot
    plot_smoothed_time_series(time_series_uns_summary_dfs, tsa_subdir, '-no-seasonality-summary-ts', plot_std=False)"""

    print('\nAnalysis complete.\n')


def main():
    """
        CLI interface for gee data analysis.
        """
    parser = argparse.ArgumentParser(
        description="process json files with network centrality measures from from GEE images")
    parser.add_argument("--input_dir",
                        help="results directory from `download_gee_data` script, containing `results_summary.json`")
    parser.add_argument('--spatial', action='store_true', default=False) # off by deafult as this takes a non-negligable amount of time

    print('-' * 35)
    print('Running analyse_gee_data.py')
    print('-' * 35)

    # parse args
    args = parser.parse_args()
    input_dir = args.input_dir
    spatial = args.spatial

    analyse_gee_data(input_dir, spatial)


if __name__ == "__main__":

    main()
