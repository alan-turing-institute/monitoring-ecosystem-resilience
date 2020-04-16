"""
Plotting code.
"""

import os
import datetime

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

import seaborn as sns

from pyveg.src.data_analysis_utils import get_AR1_parameter_estimate, get_kendell_tau, write_to_json,stl_decomposition
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plot_time_series(dfs, output_dir):
    """
    Given a dict of DataFrames, of which each row corresponds to
    a different time point (constructed with `make_time_series`),
    plot the time series of each DataFrame on the same plot.

    Parameters
    ----------
    dfs : dict of DataFrame
        The time-series results averaged over sub-locations.

    output_dir : str
        Directory to save the plot in.
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
    print(f'\nPlotting time series "{os.path.abspath(output_filename)}"...')
    plt.savefig(os.path.join(output_dir, output_filename), dpi=150)

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


def plot_smoothed_time_series(dfs, output_dir, filename_suffix ='',plot_std=True):
    """
    Given a dict of DataFrames, of which each row corresponds to
    a different time point (constructed with `make_time_series`),
    plot the time series of each DataFrame on the same plot. The
    data is assumed to have been previously smoothed, and so the 
    smoothed and unsmoothed offset50 valeus are plotted.

    Parameters
    ----------
    dfs : dict of DataFrame
        The time-series results averaged over sub-locations.

    output_dir : str
        Directory to save the plot in.
    """
    sns.set_style("white")
    for collection_name, df in dfs.items():
        if collection_name == 'COPERNICUS/S2' or 'LANDSAT' in collection_name:

            df.sort_index(inplace=True)

            # extract x values and convert to datetime objects
            try:
                veg_xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in df.index]
            except:
                # if the time series has been resampled the index is a TimeStamp object
                veg_xs = [datetime.datetime.strptime(d._date_repr,'%Y-%m-%d').date() for d in df.index]


            # extract raw means
            veg_means = df['offset50_mean']
            veg_std = df['offset50_std']

            # extract smoothed mean, std, and ci
            veg_means_smooth = df['offset50_smooth_mean']
            veg_stds_smooth = df['offset50_smooth_std']
            veg_ci = df['ci_mean']

            dfs['ECMWF/ERA5/MONTHLY'].sort_index(inplace=True)

            # extract rainfall data
            try:
                precip_xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in dfs['ECMWF/ERA5/MONTHLY'].index]
            except:
                # if the time series has been resampled the index is a TimeStamp object
                precip_xs =  [datetime.datetime.strptime(d._date_repr,'%Y-%m-%d').date() for d in dfs['ECMWF/ERA5/MONTHLY'].index]

            precip = dfs['ECMWF/ERA5/MONTHLY']['total_precipitation']

            # create a figure
            fig, ax = plt.subplots(figsize=(15,5))
            plt.xlabel('Time', fontsize=12)

            # set up veg y axis
            color = 'tab:green'
            ax.set_ylabel(f'{collection_name} Offset50', color=color, fontsize=12)
            ax.tick_params(axis='y', labelcolor=color)

            # plot unsmoothed vegetation means
            ax.plot(veg_xs, veg_means, label='Unsmoothed', linewidth=1, color='dimgray', linestyle='dotted')
            # plot LOESS smoothed vegetation means and std
            ax.plot(veg_xs, veg_means_smooth, marker='o', markersize=7, markeredgecolor=(0.9172, 0.9627, 0.9172),
                    markeredgewidth=2,
                    label='Smoothed', linewidth=2, color='green')
            if plot_std:
                # plot LOESS smoothed vegetation means and std
                ax.fill_between(veg_xs, veg_means_smooth-veg_stds_smooth, veg_means_smooth+veg_stds_smooth, facecolor='green', alpha=0.1, label='Std Dev')
            
            # plot ci of the smoothed mean
            #ax.plot(veg_xs, veg_means_smooth+veg_ci, label='99% CI', linewidth=1, color='green', linestyle='dashed')
            #ax.plot(veg_xs, veg_means_smooth-veg_ci, linewidth=1, color='green', linestyle='dashed')

            ax.set_ylim([min(veg_means)-4*max(veg_std), max(veg_means)+4*max(veg_std)])

            # plot legend
            plt.legend(loc='upper left')
            
            # duplicate x-axis for preciptation
            ax2 = ax.twinx()
            color = 'tab:blue'
            ax2.set_ylabel(f'Precipitation', color=color, fontsize=12)
            ax2.tick_params(axis='y', labelcolor=color)

            # plot precipitation
            ax2.plot(precip_xs, precip, linewidth=2, color=color, alpha=0.75)

            # add veg-precip correlation
            raw_corr = veg_means.corr(precip)
            smoothed_corr = veg_means_smooth.corr(precip)

            textstr = f'$r={smoothed_corr:.2f}$ (${raw_corr:.2f}$ unsmoothed)'
            ax2.text(0.13, 0.95, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top')

            # add autoregression info
            unsmoothed_ar1, unsmoothed_ar1_se = get_AR1_parameter_estimate(veg_means)
            smoothed_ar1, smoothed_ar1_se = get_AR1_parameter_estimate(veg_means_smooth)
            textstr = f'AR$(1)={smoothed_ar1:.2f}$ +/- ${smoothed_ar1_se:.2f}$ (${unsmoothed_ar1:.2f}$ +/- ${unsmoothed_ar1_se:.2f}$ unsmoothed)'
            ax2.text(0.45, 0.95, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top')

            ax2.set_ylim([min(precip)-3*np.array(precip).std(), max(precip)+3*np.array(precip).std()])

            # add Kendall tau
            tau, p = get_kendell_tau(veg_means)
            tau_smooth, p_smooth = get_kendell_tau(veg_means_smooth)

            # write out
            kendall_tau_dict = {}
            kendall_tau_dict['Kendall_tau'] = {'unsmoothed': {'tau': tau, 'p': p}, 'smoothed': {'tau': tau_smooth, 'p': p_smooth}}
            write_to_json(os.path.join(output_dir, collection_name.replace('/', '-')+'stats.json'), kendall_tau_dict)
            
            # add to plot
            textstr = f'$\\tau,pvalue={tau_smooth:.2f}$, ${p:.2f}$ (${tau:.2f}$, ${p_smooth:.2f}$ unsmoothed)'
            ax2.text(0.13, 0.85, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top')

            # layout
            fig.tight_layout()

            # save the plot
            output_filename = collection_name.replace('/', '-') +'-time-series-smoothed' + filename_suffix + '.png'
            print(f'\nPlotting smoothed time series "{os.path.abspath(output_filename)}"...')
            plt.savefig(os.path.join(output_dir, output_filename), dpi=150)
            plt.show()

def plot_autocorrelation_function(dfs, output_dir, filename_suffix =''):
    """
    Given a dict of DataFrames, of which each row corresponds to
    a different time point (constructed with `make_time_series`),
    plot the autocorrelation function of each DataFrame, for the 
    smoothed and unsmoothed values of offset50.

    Parameters
    ----------
    dfs : dict of DataFrame
        The time-series results averaged over sub-locations.

    output_dir : str
        Directory to save the plot in.
    """

    for collection_name, df in dfs.items():
        if collection_name == 'COPERNICUS/S2' or 'LANDSAT' in collection_name:
            
            plt.figure(figsize=(8,5))

            # make the plots
            pd.plotting.autocorrelation_plot(df['offset50_mean'], label='Unsmoothed')
            pd.plotting.autocorrelation_plot(df['offset50_smooth_mean'], label='Smoothed')
            plt.legend()

            # save the plot
            output_filename = collection_name.replace('/', '-') +'-autocorrelation-function' + filename_suffix + '.png'
            print(f'\nPlotting autocorrelation function "{os.path.abspath(output_filename)}"...')
            plt.savefig(os.path.join(output_dir, output_filename), dpi=150)

            
            # statsmodel version of the same thing
            from statsmodels.graphics.tsaplots import plot_pacf
            plot_pacf(df['offset50_mean'], label='Unsmoothed')
            plt.xlabel('Lag')
            plt.ylabel('Partial Autocorrelation')
            plt.title('Partial Autocorrelation Unsmoothed')
            plt.tight_layout()

            # save the plot
            output_filename = collection_name.replace('/', '-') +'-partial-autocorrelation-function-unsmoothed' + filename_suffix + '.png'
            print(f'\nPlotting partial autocorrelation function "{os.path.abspath(output_filename)}"...')
            plt.savefig(os.path.join(output_dir, output_filename), dpi=150)
            
            plot_pacf(df['offset50_smooth_mean'], label='Smoothed')
            plt.xlabel('Lag')
            plt.ylabel('Partial Autocorrelation')
            plt.title('Partial Autocorrelation Smoothed')
            plt.tight_layout()

            # save the plot
            output_filename = collection_name.replace('/', '-') +'-partial-autocorrelation-function-smoothed' + filename_suffix + '.png'
            print(f'\nPlotting partial autocorrelation function "{os.path.abspath(output_filename)}"...')
            plt.savefig(os.path.join(output_dir, output_filename), dpi=150)
            plt.show()


def plot_feature_vectors(dfs, output_dir):
    """
    Plot the feature vectors from the network centrality
    output of any vegetation DataFrames in `dfs`.

    Parameters
    ----------
    dfs : dict of DataFrame
        The time-series results.

    output_dir : str
        Directory to save the plot in.
    """

    for collection_name, veg_df in dfs.items():
        if collection_name == 'COPERNICUS/S2' or 'LANDSAT' in collection_name:

            # compute feature vector averaged over all sub-images
            feature_vector = np.array(veg_df.feature_vec.values.tolist()).mean(axis=0)

            # get the errors
            feature_vector_std = np.array(veg_df.feature_vec.values.tolist()).std(axis=0)

            # generate x-values
            xs = np.linspace(0,100,len(feature_vector))
            
            # make the plot
            plt.figure(figsize=(6,5))

            plt.errorbar(xs, feature_vector, marker='o', markersize=5, linestyle='', 
                         yerr=feature_vector_std, color='black', capsize=2, elinewidth=1)

            plt.xlabel('Pixel Rank (%)', fontsize=14)
            plt.ylabel('$X(V-E)$', fontsize=14)
            plt.tight_layout()

            # save the plot
            output_filename = collection_name.replace('/', '-')+'-feature-vector-summary.png'
            print(f'\nPlotting feature vector "{os.path.abspath(output_filename)}"...')
            plt.savefig(os.path.join(output_dir, output_filename), dpi=150)


            # plot also the feature vectors for different time points on the same plot
            plt.figure(figsize=(6,5))

            # loop through time points
            for _, group in veg_df.groupby('date'):
                
                # calculate feature vector
                feature_vector = np.array(group.feature_vec.values.tolist()).mean(axis=0)
                xs = np.linspace(0,100,len(feature_vector))

                # add to plot
                plt.scatter(xs, feature_vector, marker='o', color='black', alpha=0.2)

            plt.xlabel('Pixel Rank (%)', fontsize=14)
            plt.ylabel('$X(V-E)$', fontsize=14)
            plt.tight_layout()

            # save the plot
            output_filename = collection_name.replace('/', '-')+'-feature-vector-all.png'
            print(f'\nPlotting feature vector "{os.path.abspath(output_filename)}"...')
            plt.savefig(os.path.join(output_dir, output_filename), dpi=150)
            plt.show()

def plot_cross_correlations(dfs, output_dir):
    """
    Create or append the contents of `out_dict` 
    to json file `filename`.

    Parameters
    ----------
    filename: array
        Output json filename.
    out_dict: dict
        Information to save.
    """

    for collection_name, df in dfs.items():
        if collection_name == 'COPERNICUS/S2' or 'LANDSAT' in collection_name:
            
            # set up
            lags = 9
            veg_col = 'offset50_mean'            
            precip_data = dfs['ECMWF/ERA5/MONTHLY']['total_precipitation']
            correlations = []

            # make a new df to ensure NaN veg values are explicit
            df_ = pd.DataFrame()
            df_['precip'] = precip_data
            df_['offset50'] = df[veg_col]
            
            # create fig
            fig, axs = plt.subplots(3, 3, sharex='col', sharey='row', 
                                    figsize=(8, 8))

            #plt.suptitle('Precipitation vs Offset50 Lagged Scatterplot Matrix', fontsize=15, y=1.02)

            # loop through offsets
            for lag in range(0, lags):

                # select the relevant Axis object
                ax = axs.flat[lag]

                # format this subplot
                ax.set_title(f'$t-{lag}$')
                ax.grid(False)

                # plot data
                lagged_data = df_['offset50'].shift(-lag)
                
                corr = precip_data.corr(lagged_data)
                correlations.append(round(corr,4))
                sns.regplot(precip_data, lagged_data, label=f'$r={corr:.2f}$', ax=ax)
                
                # format axis label
                if lag < 6:
                    ax.set_xlabel('')
                if lag % 3 != 0:
                    ax.set_ylabel('')
                    
                ax.legend()

            plt.tight_layout()

            # save the plot
            output_filename = collection_name.replace('/', '-')+'-scatterplot-matrix.png'
            print(f'\nPlotting scatterplot matrix "{os.path.abspath(output_filename)}"...')
            plt.savefig(os.path.join(output_dir, output_filename), dpi=150)

            correlations_dict = {'lagged_correlation': correlations}
            write_to_json(os.path.join(output_dir, collection_name.replace('/', '-')+'stats.json'), correlations_dict)
            plt.show()

def stl_decomposition_plotting(ts_df,res,output_dir,output_filename):

    """
    Plot each output from the STL decomposition

     Parameters
     ----------
     ts_df : DataFrame
         The input time-series.
     res : object
        The STL fit object
     output_dir : str
         Directory to save the plot in.

    output_filename : str
         Name of the file to save the plot in.
     """


    register_matplotlib_converters()
    sns.set_style('darkgrid')
    plt.rc('figure', figsize=(20, 8))
    plt.rc('font', size=10)

    fig = res.plot()
    ax_list = fig.axes

    for ax in ax_list[:-1]:
        ax.tick_params(labelbottom=False)

    ax_list[-1].set_xticklabels(ts_df.index, rotation=45, va="center")
    plt.savefig(os.path.join(output_dir, output_filename), dpi=100)
    plt.show()

def do_stl_decomposition(dfs, period, output_dir):
    """
     Run the STL decomposition and plot the results network centrality and
     precipitation DataFrames in `dfs`.

     Parameters
     ----------
     dfs : dict of DataFrame
         The time-series results.
     peropd : float
        Periodicity to model

     output_dir : str
         Directory to save the plot in.
     """

    for collection_name, df in dfs.items():

        if 'COPERNICUS/S2' in collection_name or 'LANDSAT' in collection_name:
            offsets = df['offset50_mean']

            res = stl_decomposition(offsets, period)


            stl_decomposition_plotting(offsets,res,output_dir,collection_name.replace('/', '-')+'_STL_decomposion_'+'_offset50mean')
        elif 'ERA' in collection_name:

            precip = dfs[collection_name]['total_precipitation']   # convert to mm
            res = stl_decomposition(precip, period)

            stl_decomposition_plotting(precip,res,output_dir,collection_name.replace('/', '-')+'_STL_decomposion_'+'_precipitation')
