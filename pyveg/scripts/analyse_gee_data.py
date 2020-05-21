#!/usr/bin/env python

"""
This script analyses data previously download with `download_gee_data.py`.
First, data is preprocessed using the `analysis_preprocessing.py` module.
Plots are produced from the processed data.

"""

import os
import argparse


import pandas as pd
import ewstools
from pyveg.src.analysis_preprocessing import preprocess_data

from pyveg.src.data_analysis_utils import (
    create_lat_long_metric_figures,
    convert_to_geopandas,
    coarse_dataframe,
    moving_window_analysis,
    early_warnings_sensitivity_analysis,
    early_warnings_null_hypothesis
)

from pyveg.src.plotting import (
    plot_stl_decomposition,
    plot_feature_vector,
    plot_time_series,
    plot_ndvi_time_series,
    plot_autocorrelation_function,
    plot_cross_correlations,
    plot_moving_window_analysis,
    plot_ews_resiliance,
    plot_sensitivity_heatmap,
    kendall_tau_histograms

)


def run_time_series_analysis(filename, output_dir, detrended=False):
    """
    Make plots for the time series data. This function can
    be called for the seasonal or detrended process data.

    Parameters
    ----------
    filename : str
        Path to the time series csv to analyse.
    output_dir : str
        Path to the directory to save plots to.
    """

    # read processed data
    ts_df = pd.read_csv(filename)

    # put output plots in the results dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # auto- and cross-correlations
    # --------------------------------------------------
    # create new subdir for correlation plots
    corr_subdir = os.path.join(output_dir, 'correlations')
    if not os.path.exists(corr_subdir):
        os.makedirs(corr_subdir, exist_ok=True)

    # make autocorrelation plots
    plot_autocorrelation_function(ts_df, corr_subdir)

    # make cross correlation scatterplot matrix plots
    plot_cross_correlations(ts_df, corr_subdir)
    # --------------------------------------------------

    # time series
    # ------------------------------------------------
    # create new subdir for time series analysis
    tsa_subdir = os.path.join(output_dir, 'time-series')
    if not os.path.exists(tsa_subdir):
        os.makedirs(tsa_subdir, exist_ok=True)

    # make a smoothed time series plot
    plot_time_series(ts_df, tsa_subdir)
    plot_ndvi_time_series(ts_df, tsa_subdir)

    # plot the result of running STL decomposition
    if not detrended:
        plot_stl_decomposition(ts_df, 12, os.path.join(output_dir, 'detrended/STL'))
    # ------------------------------------------------


def run_early_warnings_resilience_analysis(filename, output_dir):
    """
    Run early warning resilience analysis on time series data. This function can
    be called on the detrended process data.

    Parameters
    ----------
    filename : str
        Path to the time series csv to analyse.
    output_dir : str
        Path to the directory to save plots to.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # read processed data
    ts_df = pd.read_csv(filename)

    # put output plots in the results dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    # old moving window analysis
    # ------------------------------------------------
    print('Running moving window analysis...')
    # create new subdir for this sub-analysis
    mwa_subdir = os.path.join(output_dir, 'moving-window')
    if not os.path.exists(mwa_subdir):
        os.makedirs(mwa_subdir, exist_ok=True)

    # run
    mwa_df = moving_window_analysis(ts_df, mwa_subdir, window_size=0.5)

    # make plots
    plot_moving_window_analysis(mwa_df, mwa_subdir)

    # save to csv
    mwa_df.to_csv(os.path.join(mwa_subdir, 'moving-window-analysis.csv'), index=False)
    # ------------------------------------------------

    # new resilience analysis
    # ------------------------------------------------
    print('Running ewstools resiliance analysis...')
    
    # create new subdir for this sub-analysis
    mwa_subdir = os.path.join(output_dir, 'ewstools')
    if not os.path.exists(mwa_subdir):
        os.makedirs(mwa_subdir, exist_ok=True)
    
    # EWS to compute (let's do all of them)
    ews = ['var', 'sd', 'ac', 'skew', 'kurt', 'ac']

    # select columns to run ews on 
    column_names = [c for c in ts_df.columns if 'offset50_mean' in c or 
                                                'ndvi_mean' in c or 
                                                'total_precipitation' in c]

    # for each relevant column
    for column_name in column_names:

        # run resilience analysis on vegetation data
        ews_dic_veg = ewstools.core.ews_compute(ts_df[column_name].dropna(),
                                    roll_window=0.5,
                                    smooth='Gaussian',
                                    lag_times=[1, 2],
                                    ews=ews,
                                    band_width=0.2)

        # make plots
        series_name = column_name.replace('_', ' ')
        plot_ews_resiliance(series_name, ews_dic_veg['EWS metrics'], ews_dic_veg['Kendall tau'], ts_df['date'], mwa_subdir)

        # sensitivity analysis
        sensitivity = early_warnings_sensitivity_analysis(ts_df[column_name].dropna(), indicators=ews)
        plot_sensitivity_heatmap(series_name, sensitivity, mwa_subdir)

        # significance tests

        significance = early_warnings_null_hypothesis(ts_df[column_name].dropna(),
                                    roll_window=0.5,
                                    smooth='Gaussian',
                                    lag_times=[1, 2],
                                    indicators=ews,
                                    band_width=0.2)

        kendall_tau_histograms(series_name, significance,mwa_subdir)


        # save results
        for key, df in ews_dic_veg.items():
            df.to_csv(os.path.join(mwa_subdir, f'ews-{column_name}__'+key.replace(' ', '')+'.csv'), index=False)


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

    # preprocess input data
    ts_dirname, dfs = preprocess_data(input_dir, n_smooth=4, resample=False, period='MS')

    # get filenames of preprocessed data time series
    ts_filenames = [f for f in os.listdir(ts_dirname) if 'time_series' in f]
    
    # put all analysis results in this dir
    output_dir = os.path.join(input_dir, 'analysis')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print('\nRunning Analysis...')
    print('-'*len('Running Analysis...'))

    # plot the feature vectors
    plot_feature_vector(output_dir)

    # for each time series
    for filename in ts_filenames:
        
        ts_file = os.path.join(ts_dirname, filename)
        print(f'\n* Analysing "{ts_file}"...')
        print('.'*50)

        # run the standard or detrended analysis
        if 'detrended' in filename:
            output_subdir = os.path.join(output_dir, 'detrended')
            run_time_series_analysis(ts_file, output_subdir, detrended=True)

            ews_subdir = os.path.join(output_dir, 'resiliance/deseasonalised')
            run_early_warnings_resilience_analysis(ts_file, ews_subdir)

        else:
            output_subdir = output_dir
            run_time_series_analysis(ts_file, output_subdir)

            ews_subdir = os.path.join(output_dir, 'resiliance/seasonal')
            run_early_warnings_resilience_analysis(ts_file, ews_subdir)

        print('.'*50, '\n')

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

    print('\nAnalysis complete.\n')


def main():
    """
    CLI interface for gee data analysis.
    """
    parser = argparse.ArgumentParser(description="process json files with network centrality measures from from GEE images")
    parser.add_argument("--input_dir", help="results directory from `download_gee_data` script, containing `results_summary.json`")
    parser.add_argument('--spatial', action='store_true', default=False) # off by deafult as this takes a non-negligable amount of time

    print('-' * 35)
    print('Running analyse_gee_data.py')
    print('-' * 35)

    # parse args
    args = parser.parse_args()
    input_dir = args.input_dir
    spatial = args.spatial

    # run analysis code
    analyse_gee_data(input_dir, spatial)


if __name__ == "__main__":
    main()
