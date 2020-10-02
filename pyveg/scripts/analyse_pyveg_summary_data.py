#!/usr/bin/env python

"""
This script analyses summary statistics produced previously with `analyse_gee_data.py` for individual locations.

"""

import os
import argparse
import json
import re
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
sns.set_style("whitegrid")

try:
    from pyveg.src.zenodo_utils import download_file, list_files, get_deposition_id
except:
    print("Unable to import zenodo_utils")


def barplot_plots(df, output_dir):

    """
    Create barplots of summary data.

    Parameters
    -----------
    df : dataframe
        Dataframe of summary data.
    output_dir : str
        Path to the directory to save plots to.
    """

    plt.figure()
    ax20 = sns.barplot(x='name',y='S2_offset50_mean_max',hue='total_precipitation_mean',data=df)
    ax20.set_xlabel("Mean precipitation over time series")
    ax20.set_ylabel("Max Offset50 over time series")
    plt.savefig(os.path.join(output_dir,'offset50_precipitation_bar.png'))

    plt.figure()
    ax20 = sns.barplot(x='name',y='S2_offset50_mean_max',hue='S2_offset50_mean_Lag-1 AC (0.99 rolling window)',data=df)
    ax20.set_xlabel("Offset50 Lag-1 AC (0.99 rolling window)")
    ax20.set_ylabel("Max Offset50 over time series")
    plt.savefig(os.path.join(output_dir,'offset50_lag1ACvalue_bar.png'))


def scatter_plots(df, output_dir):

    """
    Create scatter plots and correlation plots of summary data.

    Parameters
    -----------
    df : dataframe
        Dataframe of summary data.
    output_dir : str
        Path to the directory to save plots to.
    """

    plt.figure()
    ax = sns.scatterplot(y="S2_offset50_mean_mean", x="total_precipitation_mean", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax.set_xlabel("Mean precipitation over time series")
    ax.set_ylabel("Mean Offset50 over time series")
    plt.savefig(os.path.join(output_dir,'offset50_vs_precipitation.png'))

    plt.figure()
    ax1 = sns.scatterplot(x="longitude", y="S2_offset50_mean_mean", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax1.set_ylabel("Mean Offset50 over time series")
    ax1.set_xlabel("Longitude")
    plt.savefig(os.path.join(output_dir,'offset50_vs_Longitude.png'))

    plt.figure()
    ax2 = sns.scatterplot(x="latitude", y="S2_offset50_mean_mean", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax2.set_ylabel("Mean Offset50 over time series")
    ax2.set_xlabel("Latitude")
    plt.savefig(os.path.join(output_dir,'offset50_vs_Latitude.png'))

    plt.figure()
    ax3 = sns.scatterplot(x="longitude", y="latitude", size="S2_offset50_mean_mean", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax3.set_ylabel("Latitude")
    ax3.set_xlabel("Longitude")
    plt.savefig(os.path.join(output_dir,'lat_long_offset50.png'))

    plt.figure()
    ax4 = sns.scatterplot(y="S2_offset50_mean_Lag-1 AC (0.99 rolling window)", x="S2_offset50_mean_mean", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax4.set_ylabel("Offset50 Lag-1 AC (0.99 rolling window)")
    ax4.set_xlabel("Mean Offset50 over time series")
    plt.savefig(os.path.join(output_dir,'offset50_offset50AR1.png'))

    plt.figure()
    ax5 = sns.scatterplot(y="S2_offset50_mean_Variance (0.99 rolling window)", x="S2_offset50_mean_mean", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax5.set_ylabel("Offset50 Variance (0.99 rolling window)")
    ax5.set_xlabel("Mean Offset50 over time series")
    plt.savefig(os.path.join(output_dir,'offset50_offset50Variance.png'))

    plt.figure()
    ax6 = sns.scatterplot(y="S2_offset50_mean_Kendall tau Lag-1 AC (0.5 rolling window)", x="S2_offset50_mean_mean", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax6.set_ylabel("Offset50 Kendal tau Lag-1 AC (0.5 rolling window)")
    ax6.set_xlabel("Mean Offset50 over time series")
    plt.savefig(os.path.join(output_dir,'offset50_offset50KendaltauAR1.png'))

    plt.figure()
    ax7 = sns.scatterplot(y="S2_offset50_mean_Kendall tau Variance (0.5 rolling window)", x="S2_offset50_mean_mean", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax7.set_ylabel("Offset50 Kendal tau Variance (0.5 rolling window)")
    ax7.set_xlabel("Mean Offset50 over time series")
    plt.savefig(os.path.join(output_dir,'offset50_offset50KendaltauVariance.png'))

    plt.figure()
    ax8 = sns.scatterplot(y="S2_offset50_mean_max", x="total_precipitation_mean", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax8.set_xlabel("Mean precipitation over time series")
    ax8.set_ylabel("Max Offset50 over time series")
    plt.savefig(os.path.join(output_dir,'offset50max_vs_precipitation.png'))

    plt.figure()
    ax9 = sns.scatterplot(y="S2_offset50_mean_Lag-1 AC (0.99 rolling window)", x="S2_offset50_mean_max", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax9.set_ylabel("Offset50 Lag-1 AC (0.99 rolling window)")
    ax9.set_xlabel("Max Offset50 over time series")
    plt.savefig(os.path.join(output_dir,'offset50max_offset50AR1.png'))

    plt.figure()
    ax10 = sns.scatterplot(y="S2_offset50_mean_Variance (0.99 rolling window)", x="S2_offset50_mean_max", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax10.set_ylabel("Offset50 Variance (0.99 rolling window)")
    ax10.set_xlabel("Max Offset50 over time series")
    plt.savefig(os.path.join(output_dir,'offset50max_offset50Variance.png'))

    plt.figure()
    ax11 = sns.scatterplot(y="total_precipitation_mean", x="S2_offset50_mean_Variance (0.99 rolling window)", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax11.set_ylabel("Mean precipitation over time series")
    ax11.set_xlabel("Offset50 Variance (0.99 rolling window)")
    plt.savefig(os.path.join(output_dir,'precipitation_vs_offset50Variance.png'))

    plt.figure()
    ax12 = sns.scatterplot(y="total_precipitation_mean", x="S2_offset50_mean_Lag-1 AC (0.99 rolling window)", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax12.set_ylabel("Mean precipitation over time series")
    ax12.set_xlabel("Offset50 Lag-1 AC (0.99 rolling window)")
    plt.savefig(os.path.join(output_dir,'precipitation_vs_offset50AR1.png'))

    plt.figure()
    ax13 = sns.scatterplot(y="S2_offset50_mean_max", x="S2_ndvi_mean_max", data=df, hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax13.set_ylabel("Max offset50 over time series")
    ax13.set_xlabel("Max NDVI over time series")
    plt.savefig(os.path.join(output_dir, 'max_offset50_vs_max_NDVI.png'))

    plt.figure()
    ax14 = sns.scatterplot(y="S2_offset50_mean_mean", x="S2_ndvi_mean_mean", data=df, hue="S2_offset50_mean_pattern_type",palette="Accent_r",edgecolor="k",linewidth=1)
    ax14.set_ylabel("Mean offset50 over time series")
    ax14.set_xlabel("Mean NDVI over time series")
    plt.savefig(os.path.join(output_dir, 'mean_offset50_vs_mean_NDVI.png'))

def correlation_plots(df, output_dir):

    """
        Create correlation plots of summary data.

        Parameters
        -----------
        df : dataframe
            Dataframe of summary data.
        output_dir : str
            Path to the directory to save plots to.
        """

    #Calculate Correlations and p-values
    selected_df = df[["total_precipitation_mean","S2_offset50_mean_mean","S2_offset50_mean_max",
                      "S2_offset50_mean_Lag-1 AC (0.99 rolling window)","S2_offset50_mean_Variance (0.99 rolling window)",
                      "latitude","longitude"]]
    df_labels = ["Precipitation mean", "Ofset50 mean", "Offset50 max", "Offset50 AR1 (0.99 r.w)",
                 "Offset50 variance (0.99 r.w)", "latitude","longitude"]
    selected_corr=selected_df.corr(method="pearson")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    #Calculate p-value matrix
    def calculate_pvalues(df):
        df = df.dropna()._get_numeric_data()
        dfcols = pd.DataFrame(columns=df.columns)
        pvalues = dfcols.transpose().join(dfcols, how='outer')
        for r in df.columns:
            for c in df.columns:
                pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
        return pvalues

    p_matrix = calculate_pvalues(selected_df)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # p-value mask
    p_mask = np.invert(p_matrix < 0.05)
    p_mask_01 = np.invert(p_matrix < 0.1)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(selected_corr, cmap=cmap, vmax=1, vmin=-1, center=0, xticklabels=df_labels, yticklabels=df_labels,
                square=True, annot=True, linewidths=.5, cbar=False, cbar_kws={"shrink": .5})
    plt.savefig(os.path.join(output_dir,'pearsons_correlation_plot.png'))

    sns.heatmap(selected_corr, cmap=cmap, vmax=1, vmin=-1, center=0,xticklabels=df_labels, yticklabels=df_labels,
                square=True, annot=True, linewidths=.5, cbar=False, cbar_kws={"shrink": .5},mask=p_mask)
    plt.savefig(os.path.join(output_dir, 'significant_0.05_pearsons_correlation.png'))

    sns.heatmap(selected_corr, cmap=cmap, vmax=1, vmin=-1, center=0,xticklabels=df_labels, yticklabels=df_labels,
                square=True, annot=True, linewidths=.5, cbar=False, cbar_kws={"shrink": .5},mask=p_mask_01)
    plt.savefig(os.path.join(output_dir, 'significant_0.1_pearsons_correlation.png'))

    selected_corr.to_csv(os.path.join(output_dir,"Pearson_correlation.csv"))
    p_matrix.to_csv(os.path.join(output_dir,"Pearson_p_value.csv"))


def boxplot_plots(df, output_dir):
    """
        Create boxplots of summary data.

        Parameters
        -----------
        df : dataframe
            Dataframe of summary data.
        output_dir : str
            Path to the directory to save plots to.
        """

    plt.figure()
    ax15 = sns.boxplot(x="S2_offset50_mean_pattern_type", y="S2_offset50_mean_max", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r")
    ax15.set_ylabel("Max Offset50 Values")
    ax15.set_xlabel("Pattern Type")
    plt.savefig(os.path.join(output_dir, 'Max_offset50_boxplot.png'))

    plt.figure()
    ax16 = sns.boxplot(x="S2_offset50_mean_pattern_type", y="S2_offset50_mean_mean", data=df,hue="S2_offset50_mean_pattern_type",palette="Accent_r")
    ax16.set_ylabel("Mean Offset50 Values")
    ax16.set_xlabel("Pattern Type")
    plt.savefig(os.path.join(output_dir, 'Mean_offset50_boxplot.png'))


def process_input_data(input_location):

    """Read all input summary statistics and transform data into
     a more analysis-friendly format

    Parameters
    ----------
    input_location : str
        Location of summary statistics output files from analyse_gee_data.py.
        Can be 'zenodo' or 'zenodo_test' to download from the production or sandbox Zenodo depository,
        or the path to a directory on the local filesystem.

    Returns
    -------
    df:  DataFrame containing all the time-series data concatenated together.
    """
    df_list = []
    if input_location == 'zenodo' or input_location == 'zenodo_test':
        is_test = input_location == 'zenodo_test'
        deposition_id = get_deposition_id(is_test)
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_filenames = [download_file(f, deposition_id, tmpdir, is_test) \
                            for f in list_files(deposition_id, "csv", is_test)]
            df_list = [pd.read_csv(filename) for filename in ts_filenames]
    else:
        ts_filenames = [f for f in os.listdir(input_location) if "_summary_stats" in f]
        df_list = [pd.read_csv(os.path.join(input_location, filename)) for filename in ts_filenames]

    df = pd.concat(df_list)

    ts_dict_list = []
    for name, group in df.groupby(["latitude", "longitude"]):

        dict_list = {}

        for ts in np.unique(group['ts_id']):
            for col in group:
                dict_list[ts + "_" + col] = group[group['ts_id'] == ts][col].values[0]
        dict_list['name'] = name
        dict_list['latitude'] = group['latitude'].values[0]
        dict_list['longitude'] = group['longitude'].values[0]

        ts_dict_list.append(dict_list)

    return pd.DataFrame(ts_dict_list)



def analyse_pyveg_summary_data(input_location, output_dir):

    """
    Run analysis on summary statistics data

    Parameters
    ----------
    input_location : str
        Location of summary statistics output files from analyse_gee_data.py.
        Can be 'zenodo' or 'zenodo_test' for the production or sandbox Zenodo depositions, or
        the path to the local directory containing the files.
     output_dir: str,
        Location for outputs of the analysis. If None, use input_dir
    """

    df = process_input_data(input_location)

    summary_plots = os.path.join(output_dir, "summary_plots")
    if not os.path.exists(summary_plots):
        os.makedirs(summary_plots, exist_ok=True)

    scatter_plots(df,summary_plots)
    barplot_plots(df,summary_plots)
    correlation_plots(df,summary_plots)
    boxplot_plots(df,summary_plots)


def main():
    """
    CLI interface for gee data analysis.
    """
    parser = argparse.ArgumentParser(
        description="process json files with network centrality measures from from GEE images"
    )
    parser.add_argument(
        "--input_location",
        help="results directory from `download_gee_data` script, containing `summary_stats.csv`, OR 'zenodo' or 'zenodo_test' to download the files from the production or sandbox Zenodo depositions.",
    )
    parser.add_argument(
        "--output_dir",
        help="location where analysis plots will be put.",default="."
    )


    print("-" * 35)
    print("Running analyse_pyveg_summary_data.py")
    print("-" * 35)

    # parse args
    args = parser.parse_args()


    # run analysis code
    analyse_pyveg_summary_data(args.input_location,
                               args.output_dir)


if __name__ == "__main__":
    main()
