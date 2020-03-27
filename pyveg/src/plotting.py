"""
Plotting code.
"""

import os
import datetime

import numpy as numpy
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm


def plot_time_series(dfs, output_dir):
    #
    """
    Given a dict of DataFrames, of which each row corresponds to
    a different time point (constructed with `make_time_series`),
    plot the time series of each DataFrame on the same plot.

    Parameters
    ----------
    dfs : dict of DataFrame
        The time-series results averaged over sub-locations.
    """

    # function to help plot many y axes
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # setup plot
    fig, ax1 = plt.subplots(figsize=(15,5))
    fig.subplots_adjust(right=0.9)

    # set up x axis to handle dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    ax1.set_xlabel('Time')

    #print(get_weather_time_series(dfs))
    #print(get_veg_time_series(dfs))

    """
    for collection_name, df in dfs.items():

        if 'offset50' in df.columns:
            # prepare data
            dates = df.index
            xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
            means = df['offset50']
            stds = df['offset50_std']
        else: # assume
            # prepare data
            dates = df.index
            xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
            print(df.values)
            # if there are multiple data columns, use them all
            ys_list = []

        # instantiate a new shared axis
        ax2 = ax1.twinx()
    """

    s2 = 'COPERNICUS/S2'
    l8 = 'LANDSAT/LC08/C01/T1_SR'

    # prepare data
    cop_means = dfs[s2]['offset50_mean']
    cop_stds = dfs[s2]['offset50_std']
    cop_dates = dfs[s2].index
    cop_xs = [datetime.datetime.strptime(str(d),'%Y-%m-%d').date() for d in cop_dates]

    #l8_means = dfs[l8]['offset50']
    #l8_stds = dfs[l8]['offset50_std']
    #l8_dates = dfs[l8].index
    #l8_xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in l8_dates]

    precip = dfs['ECMWF/ERA5/MONTHLY']['total_precipitation'] * 1000 # convert to mm
    temp = dfs['ECMWF/ERA5/MONTHLY']['mean_2m_air_temperature'] - 273.15 # convert to Celcius
    weather_dates = dfs['ECMWF/ERA5/MONTHLY'].index
    w_xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in weather_dates]

    # add copernicus
    color = 'tab:green'
    ax1.set_ylabel('Copernicus Offset50', color=color)
    ax1.plot(cop_xs, cop_means, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([-900, -400])
    plt.fill_between(cop_xs, cop_means-cop_stds, cop_means+cop_stds,
                     facecolor='green', alpha=0.1)

    # add precip
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Precipitation [mm]', color=color)  # we already handled the x-label with ax1
    ax2.set_ylim([-10, 250])
    ax2.plot(w_xs, precip, color=color, alpha=0.5, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    # add temp
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.075))
    make_patch_spines_invisible(ax3)
    ax3.spines["right"].set_visible(True)
    ax3.set_ylim([22, 36])
    color = 'tab:red'
    ax3.set_ylabel('Mean Temperature [$^\circ$C]', color=color)  # we already handled the x-label with ax1
    ax3.plot(w_xs, temp, color=color, alpha=0.2, linewidth=2)
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # save the plot before adding Landsat
    output_filename = 'time-series.png'
    plt.savefig(os.path.join(output_dir, output_filename), dpi=100)

    # add l8
    #ax4 = ax1.twinx()
    #ax4.spines["left"].set_position(("axes", -0.1))
    #ax4.spines["left"].set_visible(True)
    #make_patch_spines_invisible(ax4)
    #color = 'tab:purple'
    #ax4.set_ylabel('landsat', color=color)  # we already handled the x-label with ax1
    #ax4.plot(l8_xs, l8_means, color=color)
    #ax4.tick_params(axis='y', labelcolor=color)
    #ax4.yaxis.tick_left()
    #plt.fill_between(l8_xs, l8_means-l8_stds, l8_means+l8_stds,
    #                 facecolor='purple', alpha=0.05)

    # save the plot
    #output_filename = 'time-series-full.png'
    #plt.savefig(os.path.join(output_dir, output_filename), dpi=100)

    """# ------------------------------------------------
    # setup plot
    fig, ax1 = plt.subplots(figsize=(13,5))
    fig.subplots_adjust(right=0.9)

    # set up x axis to handle dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    ax1.set_xlabel('Time')

    # add copernicus
    color = 'tab:green'
    ax1.set_ylabel('Copernicus Offset50', color=color)
    ax1.plot(cop_xs, cop_means, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.fill_between(cop_xs, cop_means-cop_stds, cop_means+cop_stds,
                     facecolor='green', alpha=0.2)

    # add l8
    ax4 = ax1.twinx()
    color = 'tab:purple'
    ax4.set_ylabel('landsat', color=color)  # we already handled the x-label with ax1
    #ax4.yaxis.tick_left()
    ax4.plot(l8_xs, l8_means, color=color)
    ax4.tick_params(axis='y', labelcolor=color)
    plt.fill_between(l8_xs, l8_means-l8_stds, l8_means+l8_stds,
                     facecolor='purple', alpha=0.2)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # save the plot
    output_filename = 'time-series-offsets-only.png'
    plt.savefig(os.path.join(output_dir, output_filename), dpi=100)
    """


def plot_smoothed_time_series(dfs, output_dir):

    for collection_name, df in dfs.items():
        if collection_name == 'COPERNICUS/S2' or 'LANDSAT' in collection_name:

            # extract x values and convert to datetime objects
            veg_xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in df.index]

            # extract raw means
            veg_means = df['offset50_mean']

            # extract smoothed mean, std, and ci
            veg_means_smooth = df['offset50_smooth_mean']
            veg_stds_smooth = df['offset50_smooth_std']
            veg_ci = df['ci_mean']

            # extract rainfall data
            precip_xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in dfs['ECMWF/ERA5/MONTHLY'].index]
            precip = dfs['ECMWF/ERA5/MONTHLY']['total_precipitation']

            # create a figure
            fig, ax = plt.subplots(figsize=(15,5))
            plt.xlabel('Time', fontsize=12)

            # set up veg y axis
            color = 'tab:green'
            ax.set_ylabel(f'{collection_name} Offset50', color=color, fontsize=12)
            ax.tick_params(axis='y', labelcolor=color)

            # plot unsmoothed vegetation means
            ax.plot(veg_xs, veg_means, label='Raw', linewidth=1, color='dimgray', linestyle='dotted')
            
            # plot LOESS smoothed vegetation means and std
            ax.plot(veg_xs, veg_means_smooth, label='Smoothed', linewidth=2, color='green')
            ax.fill_between(veg_xs, veg_means_smooth-veg_stds_smooth, veg_means_smooth+veg_stds_smooth, facecolor='green', alpha=0.1, label='Std Dev')
            
            # plot ci of the smoothed mean
            #ax.plot(veg_xs, veg_means_smooth+veg_ci, label='99% CI', linewidth=1, color='green', linestyle='dashed')
            #ax.plot(veg_xs, veg_means_smooth-veg_ci, linewidth=1, color='green', linestyle='dashed')
            
            # plot legend
            plt.legend(loc='upper left')
            
            # duplicate x-axis for preciptation
            ax2 = ax.twinx()
            color = 'tab:blue'
            ax2.set_ylabel(f'Precipitation', color=color, fontsize=12)
            ax2.tick_params(axis='y', labelcolor=color)

            # plot precipitation
            ax2.plot(precip_xs, precip, linewidth=2, color=color, alpha=0.75)

            raw_corr = veg_means.corr(precip)
            smoothed_corr = veg_means_smooth.corr(precip)
            textstr = f'r={round(smoothed_corr, 2)} ({round(raw_corr, 2)})'
            ax2.text(0.12, 0.95, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top')

            # layot
            fig.tight_layout()

            # save the plot before adding Landsat
            output_filename = collection_name.replace('/', '-')+'-time-series-smoothed.png'
            plt.savefig(os.path.join(output_dir, output_filename), dpi=100)
