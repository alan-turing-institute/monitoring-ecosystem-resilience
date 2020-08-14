#!/usr/bin/env python

"""
This script analyses summary statistics produced previously with `analyse_gee_data.py` for individual locations.

"""

import os
import argparse
import json
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

from pyveg.scripts.upload_to_zenodo import upload_results

def process_input_data(input_dir):

    """Read all input summary statistics and transform data into
     a more analysis-friendly format

    Parameters
    ----------
    input_dir : str
        Location of summary statistics output files from analyse_gee_data.py
    """

    ts_filenames = [f for f in os.listdir(input_dir) if "time_series_summary_stats" in f]

    df_list = [pd.read_csv(os.path.join(input_dir, filename)) for filename in ts_filenames]
    df = pd.concat(df_list)

    ts_dict_list = []
    for name, group in df.groupby(["latitude", "longitude"]):

        dict_list = {}

        for ts in np.unique(group['ts_id']):
            for col in group:
                dict_list[ts + "_" + col] = group[group['ts_id'] == ts][col].values[0]
        dict_list['name'] = name

        ts_dict_list.append(dict_list)

    return pd.DataFrame(ts_dict_list)



def analyse_pyveg_summary_data(input_dir, output_dir):

    """
    Run analysis on summary statistics data

    Parameters
    ----------
    input_dir : str
        Location of summary statistics output files from analyse_gee_data.py
     output_dir: str,
        Location for outputs of the analysis. If None, use input_dir
    """

    df = process_input_data(input_dir)

    variables =  ['S2_ndvi_mean_mean','S2_offset50_mean_mean','total_precipitation_mean', 'S2_offset50_mean_Lag-1 AC (0.99 rolling window)','S2_offset50_mean_Variance (0.99 rolling window)',]

    g = sns.pairplot(df[variables],corner=True)
    g.fig.set_size_inches(10, 10)

    print (df)
    return 0


def main():
    """
    CLI interface for gee data analysis.
    """
    parser = argparse.ArgumentParser(
        description="process json files with network centrality measures from from GEE images"
    )
    parser.add_argument(
        "--input_dir",
        help="results directory from `download_gee_data` script, containing `results_summary.json`",
    )
    parser.add_argument(
        "--output_dir",
        help="location where analysis plots will be put.  If not specified, will use input_dir",
    )


    print("-" * 35)
    print("Running analyse_pyveg_summary_data.py")
    print("-" * 35)

    # parse args
    args = parser.parse_args()
    # check we have the bare minimum of args set that we need
    output_dir = args.output_dir if args.output_dir else args.input_dir
    if not output_dir:
        raise RuntimeError("Need to specify --output_dir argument if reading from Azure blob storage")

    input_location = args.input_dir

    # run analysis code
    analyse_pyveg_summary_data(input_location,
                     output_dir)


if __name__ == "__main__":
    main()
