#!/usr/bin/env python

"""
Scripts to process the output of the GEE images and json files with network centrality measures.

The json outputs are turned into a dataframe and the values of a particular metric a plotted
as a function of time.

Finally a GIF file is produced with all of the network metric images, as well as the original 10km x 10km dowloaded images.
"""

import argparse
import os

from pyveg.src.data_analysis_utils import (
    variable_read_json_to_dataframe,
    drop_veg_outliers,
    smooth_veg_data,
    make_time_series,
    create_lat_long_metric_figures,
    convert_to_geopandas,
    coarse_dataframe,
    write_slimmed_csv,
    remove_seasonality_combined,
    remove_seasonality_all_sub_images,
    variance_moving_average_time_series,
)
from pyveg.src.plotting import (
    do_stl_decomposition,
    plot_smoothed_time_series,
    plot_autocorrelation_function,
    plot_feature_vectors,
    plot_cross_correlations
)


def analyse_gee_data(input_dir, do_spatial_plot, do_time_series_plot):

    """
    Run analysis on dowloaded gee data

    Parameterss
    ----------
    input_dir : string
        Path to directory with downloaded dada
    do_spatial_plot: bool
        Option to run spatial analysis and do plots
    do_time_series_plot: bool
        Option to run time-series analysis and do plots

    """

    # put output plots in the results dir
    output_dir = os.path.join(input_dir, 'analysis')

    # check input file exists
    json_summary_path = os.path.join(input_dir, 'results_summary.json')
    if not os.path.exists(json_summary_path):
        raise FileNotFoundError(f'Could not find file "{os.path.abspath(json_summary_path)}".')

    # make output subdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # read all json files in the directory and produce a dataframe
    print(f"Reading results from '{os.path.abspath(json_summary_path)}'...")
    dfs = variable_read_json_to_dataframe(json_summary_path)

    # spatial analysis and plotting
    # ------------------------------------------------
    if do_spatial_plot:

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

    # time series analysis and plotting
    # ------------------------------------------------
    if do_time_series_plot:

        # create new subdir for time series analysis
        # tsa_subdir = os.path.join(output_dir, 'time-series') # if we start to have more and more results
        tsa_subdir = output_dir

        if not os.path.exists(tsa_subdir):
            os.makedirs(tsa_subdir, exist_ok=True)

        # convert to time series
        time_series_dfs = make_time_series(dfs.copy())

        # make the old time series plot
        # print('\nPlotting time series...')
        # plot_time_series(time_series_dfs, tsa_subdir)

        # remove outliers from the time series
        dfs = drop_veg_outliers(dfs, sigmas=3)  # not convinced this is really helping much

        # plot the feature vectors averaged over all time points and sub images
        try:
            plot_feature_vectors(dfs, tsa_subdir)
        except AttributeError:
            print('Can not plot feature vectors...')

            # LOESS smoothing on sub-image time series
        smoothed_time_series_dfs = make_time_series(smooth_veg_data(dfs.copy(), n=4))  # increase smoothing with n>5

        # make a smoothed time series plot
        plot_smoothed_time_series(smoothed_time_series_dfs, tsa_subdir)

        # make autocorrelation plots
        plot_autocorrelation_function(smoothed_time_series_dfs, tsa_subdir)

        # make cross correlation scatterplot matrix plots
        plot_cross_correlations(smoothed_time_series_dfs, tsa_subdir)

        # write csv for easy external analysis
        write_slimmed_csv(smoothed_time_series_dfs, tsa_subdir)

        #---------------------------------------------------

        new_series = variance_moving_average_time_series(dfs['COPERNICUS/S2'],'offset50',dfs['COPERNICUS/S2'].shape[0]/2)

        # ------------------------------------------------

        do_stl_decomposition(time_series_dfs, 12, tsa_subdir)

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
        plot_smoothed_time_series(time_series_uns_summary_dfs, tsa_subdir, '-no-seasonality-summary-ts', plot_std=False)

    print('\nDone!\n')

def main():
    """
        CLI interface for gee data analysis.
        """
    parser = argparse.ArgumentParser(
        description="process json files with network centrality measures from from GEE images")
    parser.add_argument("--input_dir",
                        help="results directory from `download_gee_data` script, containing `results_summary.json`")
    parser.add_argument('--spatial_plot', action='store_true')
    parser.add_argument('--time_series_plot', action='store_true', default=True)

    print('-' * 35)
    print('Running analyse_gee_data.py')
    print('-' * 35)

    # parse args
    args = parser.parse_args()
    input_dir = args.input_dir

    analyse_gee_data(input_dir, args.spatial_plot, args.time_series_plot)


if __name__ == "__main__":

    main()
