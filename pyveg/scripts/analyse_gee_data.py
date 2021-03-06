#!/usr/bin/env python

"""
This script analyses data previously download with `download_gee_data.py`.
First, data is preprocessed using the `analysis_preprocessing.py` module.
Plots are produced from the processed data.

"""

import os
import argparse
import json
import re

import pandas as pd
import ewstools

from pyveg.src.analysis_preprocessing import (
    read_results_summary,
    preprocess_data,
    save_ts_summary_stats
)

from pyveg.src.data_analysis_utils import (
    create_lat_long_metric_figures,
    convert_to_geopandas,
    coarse_dataframe,
    moving_window_analysis,
    early_warnings_sensitivity_analysis,
    early_warnings_null_hypothesis,
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
    plot_correlation_mwa,
    kendall_tau_histograms,
)

from pyveg.scripts.create_analysis_report import create_markdown_pdf_report
from pyveg.scripts.upload_to_zenodo import upload_summary_stats

# if time-series is fewer than 12 points, can't do Early Warning Signals analysis
MIN_TS_SIZE_FOR_EWS = 12

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
    corr_subdir = os.path.join(output_dir, "correlations")
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
    tsa_subdir = os.path.join(output_dir, "time-series")
    if not os.path.exists(tsa_subdir):
        os.makedirs(tsa_subdir, exist_ok=True)

    # make a smoothed time series plot
    plot_time_series(ts_df, tsa_subdir)
    plot_ndvi_time_series(ts_df, tsa_subdir)

    # plot the result of running STL decomposition
    if not detrended:
        plot_stl_decomposition(ts_df,  MIN_TS_SIZE_FOR_EWS, os.path.join(output_dir, "detrended/STL"))
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
    print("Running moving window analysis...")
    # create new subdir for this sub-analysis
    mwa_subdir = os.path.join(output_dir, "moving-window")
    if not os.path.exists(mwa_subdir):
        os.makedirs(mwa_subdir, exist_ok=True)

    # run
    mwa_df = moving_window_analysis(ts_df, mwa_subdir, window_size=0.5)

    # make plots
    plot_moving_window_analysis(mwa_df, mwa_subdir)
    plot_correlation_mwa(mwa_df, mwa_subdir)

    # save to csv
    mwa_df.to_csv(os.path.join(mwa_subdir, "moving-window-analysis.csv"), index=False)
    # ------------------------------------------------

    # new resilience analysis
    # ------------------------------------------------
    print("Running ewstools resiliance analysis...")

    # create new subdir for this sub-analysis
    mwa_subdir = os.path.join(output_dir, "ewstools")
    if not os.path.exists(mwa_subdir):
        os.makedirs(mwa_subdir, exist_ok=True)

    # EWS to compute (let's do all of them)
    ews = ["var", "sd", "ac", "skew", "kurt", "ac"]

    # select columns to run ews on
    column_names = [
        c
        for c in ts_df.columns
        if "offset50_mean" in c or "ndvi_mean" in c or "total_precipitation" in c
    ]

    # for each relevant column
    for column_name in column_names:

        # run resilience analysis on vegetation data

        ews_dic_veg = ewstools.core.ews_compute(ts_df[column_name].dropna(),
                                    roll_window=0.5,
                                    smooth='Gaussian',
                                    lag_times=[1, 2],
                                    ews=ews,
                                    band_width=6)

        # make plots
        series_name = column_name.replace("_", " ")
        plot_ews_resiliance(
            series_name,
            ews_dic_veg["EWS metrics"],
            ews_dic_veg["Kendall tau"],
            ts_df["date"],
            mwa_subdir,
        )

        # sensitivity analysis
        sensitivity = early_warnings_sensitivity_analysis(
            ts_df[column_name].dropna(), indicators=ews
        )
        plot_sensitivity_heatmap(series_name, sensitivity, mwa_subdir)

        # significance tests

        significance = early_warnings_null_hypothesis(ts_df[column_name].dropna(),
                                    roll_window=0.5,
                                    smooth='Gaussian',
                                    lag_times=[1, 2],
                                    indicators=ews,
                                    band_width=6)

        kendall_tau_histograms(series_name, significance, mwa_subdir)

        # save results
        for key, df in ews_dic_veg.items():
            df.to_csv(
                os.path.join(
                    mwa_subdir, f"ews-{column_name}__" + key.replace(" ", "") + ".csv"
                ),
                index=False,
            )


def analyse_gee_data(input_location,
                     input_location_type="local",
                     input_json_file=None,
                     output_dir=None,
                     do_time_series=True,
                     do_spatial=False,
                     upload_to_zenodo=False,
                     upload_to_zenodo_test=False):
    """
    Run analysis on dowloaded gee data

    Parameters
    ----------
    input_location : str
        Location of results_summary.json output from pyveg_run_pipeline,
        OR if input_location_type is `zenodo` or `zenodo_test`, the 2-digit coordinate_id
        representing the row in `coordinates.py`.
    input_location_type: str
        Can be 'local', 'azure', 'zenodo', or 'zenodo_test'.
    input_json: str, optional. Full path to the results summary json file.
    output_dir: str,
        Location for outputs of the analysis. If None, use input_location
    do_time_series: bool
        Option to run time-series analysis and do plots
    do_spatial: bool
        Option to run spatial analysis and do plots
    upload_to_zenodo: bool
        Upload results to the production Zenodo repo
    upload_to_zenodo_test: bool
        Upload results to the sandbox Zenodo repo
    """

    if not output_dir:
        output_dir = input_location
    # read the results_summary.json
    if input_json_file:
        input_json = json.load(open(input_json_file))
    else:
        input_json = read_results_summary(input_location,
                                          input_location_type=input_location_type)
    # preprocess input data
    ts_dirname, dfs = preprocess_data(
        input_json, output_dir, n_smooth=4, resample=False, period="MS"
    )

    # get filenames of preprocessed data time series
    ts_filenames = [f for f in os.listdir(ts_dirname) if "time_series" in f]

    # put all analysis results in this dir
    output_analysis_dir = os.path.join(output_dir, "analysis")
    if not os.path.exists(output_analysis_dir):
        os.makedirs(output_analysis_dir, exist_ok=True)

    print("\nRunning Analysis...")
    print("-" * len("Running Analysis..."))

    # plot the feature vectors
    plot_feature_vector(output_analysis_dir)

    # time-series analysis and plotting
    # check first if data is a time series
    ts_df = pd.read_csv(os.path.join(ts_dirname,ts_filenames[0]))
    size_ts = ts_df.shape[0]
    if size_ts <= 2:
        print ('WARNING: Less than 3 times points, not possible to do a time series analysis')
        do_time_series = False
    # -----------------------------------


    # for each time series
    if do_time_series:

        # put output plots in the results dir
        input_dir_ts = os.path.join(output_dir, "processed_data")

        save_ts_summary_stats(input_dir_ts, output_analysis_dir,input_json['metadata'])

        for filename in ts_filenames:

            ts_file = os.path.join(ts_dirname, filename)
            print(f'\n* Analysing "{ts_file}"...')
            print("." * 50)

            # run the standard or detrended analysis
            if "detrended" in filename:
                output_subdir = os.path.join(output_analysis_dir, "detrended")
                run_time_series_analysis(ts_file, output_subdir, detrended=True)

                # resilience analysis only done in large enough time series
                if size_ts > MIN_TS_SIZE_FOR_EWS:
                    ews_subdir = os.path.join(output_analysis_dir, "resiliance/deseasonalised")
                    run_early_warnings_resilience_analysis(ts_file, ews_subdir)

            else:
                output_subdir = output_analysis_dir
                run_time_series_analysis(ts_file, output_subdir)

                # resilience analysis only done in large enough time series
                if size_ts > MIN_TS_SIZE_FOR_EWS:
                    ews_subdir = os.path.join(output_analysis_dir, "resiliance/seasonal")
                    run_early_warnings_resilience_analysis(ts_file, ews_subdir)

            print("." * 50, "\n")

    # spatial analysis and plotting
    # ------------------------------------------------
    if do_spatial:

        # from the dataframe, produce network metric figure for each avalaible date
        print("\nCreating spatial plots...")

        # create new subdir for time series analysis
        spatial_subdir = os.path.join(output_analysis_dir, "spatial")
        if not os.path.exists(spatial_subdir):
            os.makedirs(spatial_subdir, exist_ok=True)

        for collection_name, df in dfs.items():
            if collection_name == "COPERNICUS/S2" or "LANDSAT" in collection_name:
                data_df_geo = convert_to_geopandas(df.copy())
                data_df_geo_coarse = coarse_dataframe(data_df_geo.copy(), 2)
                create_lat_long_metric_figures(
                    data_df_geo_coarse, "offset50", spatial_subdir
                )
    # ------------------------------------------------

    print('\nCreating report.\n')

    for collection_name, df in dfs.items():
        if collection_name == 'COPERNICUS/S2' or 'LANDSAT' in collection_name:

            try:
                metadata = input_json["metadata"] if "metadata" in input_json.keys() else None

                if input_location_type=='local':
                    from pathlib import Path

                    parent_path = Path(input_location).parent
                    rgb_location = parent_path
                else:
                    rgb_location = input_location
                create_markdown_pdf_report(output_dir,
                                           "local",
                                           rgb_location,
                                           input_location_type,
                                           do_time_series,
                                           output_dir,
                                           collection_name,
                                           metadata)

            except Exception as e:
                print ("Warning: A problem was found, the report was not created. There might be missing figures needed "
                              "for the report or a problem with the pandoc installation. {}".format(e))

    # ------------------------------------------------

    # ------------------------------------------------
    # upload the summary csv file to Zenodo
    if upload_to_zenodo or upload_to_zenodo_test:
        print('\nUploading results to Zenodo.\n')
        analysis_dir = os.path.join(output_dir, "analysis")
        filenames = [f for f in os.listdir(analysis_dir) if f.endswith(".csv")
                     and f != "time_series_summary_stats.csv"]
        if filenames:
            filepath = os.path.join(analysis_dir, filenames[0])



            uploaded = upload_summary_stats(filepath,
                                            upload_to_zenodo_test)
            if uploaded:
                print("Uploaded {} to Zenodo.".format(filenames[0]))
        else:
            print("Couldn't find time series summary stats csv file. Not uploading to Zenodo.")


    # ------------------------------------------------

    print("\nAnalysis complete.\n")


def main():
    """
    CLI interface for gee data analysis.
    """
    parser = argparse.ArgumentParser(
        description="process json files with network centrality measures from from GEE images"
    )
    parser.add_argument(
        "--input_json",
        help="path to results file from `pyveg_run_pipeline` command.  Use this  OR '--input_dir' OR '--input_container.",
    )
    parser.add_argument(
        "--input_dir",
        help="results directory from `pyveg_run_pipeline` command, containing `results_summary.json`",
    )
    parser.add_argument(
        "--input_container",
        help="results location on blob storage from `pyveg_run_pipeline` command, containing `results_summary.json`",
    )
    parser.add_argument(
        "--input_zenodo_coords",
        help="If results_summary json is uploaded to Zenodo deposition, give the two digit coordinate id from coordinates.py, e.g. '00'",
    )
    parser.add_argument(
        "--input_zenodo_test_coords",
        help="If results_summary json is uploaded to Zenodo sandbox deposition, give the two digit coordinate id from coordinates.py, e.g. '00'",
    )
    parser.add_argument(
        "--output_dir",
        help="location where analysis plots will be put.  If not specified, will use input_dir",
    )
    parser.add_argument(
        "--dont_do_time_series", action="store_true", default=False
    )  # if set, disable the time-series analysis

    parser.add_argument(
        "--spatial", action="store_true", default=False
    )  # off by default as this takes a non-negligable amount of time

    parser.add_argument(
        "--upload_to_zenodo", help="store the summary_stats.csv file on Zenodo", action="store_true", default=False
    )  # off by deafult

    parser.add_argument(
        "--upload_to_zenodo_test", help="store the summary_stats.csv file on Zenodo sandbox", action="store_true", default=False
    )  # off by deafult

    print("-" * 35)
    print("Running analyse_gee_data.py")
    print("-" * 35)

    # parse args
    args = parser.parse_args()
    # check we have the bare minimum of args set that we need
    output_dir = args.output_dir if args.output_dir else args.input_dir
    if not output_dir:
        raise RuntimeError("Need to specify --output_dir argument if reading from Azure blob storage or Zenodo")

    # read the input json, using either input_dir or input_container arguments
    if args.input_json and (args.input_dir or args.input_container):
        raise RuntimeError("""
        Please use only one of --input_dir or --input_json (for local input), or --onput_container (for Azure),
        or --zenodo_coords_id (for production Zenodo deposition) or --zenodo_test_coords_id (for Zenodo sandbox)
        """)
    elif args.input_dir and args.input_container:
        raise RuntimeError("""
        Please use only one of --input_dir or --input_json (for local input), or --onput_container (for Azure),
        or --zenodo_coords_id (for production Zenodo deposition) or --zenodo_test_coords_id (for Zenodo sandbox)
        """)
    elif (args.input_dir or args.input_json or args.input_container) and \
         (args.input_zenodo_coords or args.input_zenodo_test_coords):
        raise RuntimeError("""
        Please use only one of --input_dir or --input_json (for local input), or --onput_container (for Azure),
        or --zenodo_coords_id (for production Zenodo deposition) or --zenodo_test_coords_id (for Zenodo sandbox)
        """)
    if args.input_container:
        input_location = args.input_container
        input_location_type = "azure"
    elif args.input_zenodo_coords:
        input_location = args.input_zenodo_coords
        input_location_type = "zenodo"
    elif args.input_zenodo_test_coords:
        input_location = args.input_zenodo_test_coords
        input_location_type = "zenodo_test"
    else:
        input_location_type = "local"
        if args.input_dir:
            input_location = args.input_dir
        else:
            input_location = None
    input_json = args.input_json

    do_time_series = not args.dont_do_time_series
    do_spatial = args.spatial
    upload_to_zenodo = args.upload_to_zenodo
    upload_to_zenodo_test = args.upload_to_zenodo_test
    # run analysis code
    analyse_gee_data(input_location,
                     input_location_type,
                     input_json,
                     output_dir,
                     do_time_series,
                     do_spatial,
                     upload_to_zenodo,
                     upload_to_zenodo_test)


if __name__ == "__main__":
    main()
