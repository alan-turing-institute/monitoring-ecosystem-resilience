"""
Plotting code.
"""

import datetime
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

from pyveg.src.data_analysis_utils import get_AR1_parameter_estimate, get_kendell_tau, write_to_json, stl_decomposition

register_matplotlib_converters()

# globally set image quality
plot_dpi = 150


def plot_time_series(df, output_dir, filename_suffix =''):
    """
    Given a time series DataFrames (constructed with `make_time_series`),
    plot the vegetitation and precipitation time series.

    Parameters
    ----------
    df : DataFrame
        Time series DataFrame.

    output_dir : str
        Directory to save the plots in.
    """

    def make_plot(df, veg_prefix, output_dir, veg_prefix_b=None):

        # handle the case where vegetation and precipitation have mismatched NaNs
        veg_df = df.dropna(subset=[veg_prefix+'_offset50_mean'])

        # get vegetation x values to datetime objects
        try:
            veg_xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in veg_df.date]
        except:
            # if the time series has been resampled the index is a TimeStamp object
            veg_xs = [datetime.datetime.strptime(d._date_repr,'%Y-%m-%d').date() for d in veg_df.date]

        # get vegetation y values
        veg_means = veg_df[veg_prefix+'_offset50_mean']
        veg_std = veg_df[veg_prefix+'_offset50_std']

        # create a figure
        fig, ax = plt.subplots(figsize=(15, 4.5))
        plt.xlabel('Time', fontsize=14)

        # set up veg y axis
        color = 'tab:green'
        ax.set_ylabel(f'{veg_prefix} Offset50', color=color, fontsize=14)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim([veg_means.min() - 1*veg_std.max(), veg_means.max() + 3*veg_std.max()])

        # plot unsmoothed vegetation means
        ax.plot(veg_xs, veg_means, label='Unsmoothed', linewidth=1, color='dimgray', linestyle='dotted')

        # add smoothed time series if availible
        if any(['smooth' in c and veg_prefix in c for c in veg_df.columns]):

            # get smoothed mean, std
            veg_means_smooth = veg_df[veg_prefix+'_offset50_smooth_mean']
            veg_stds_smooth = veg_df[veg_prefix+'_offset50_smooth_std']

            # plot smoothed vegetation means and std
            ax.plot(veg_xs, veg_means_smooth, marker='o', markersize=7, markeredgecolor=(0.9172, 0.9627, 0.9172),
                    markeredgewidth=2,
                    label='Smoothed', linewidth=2, color='green')

            ax.fill_between(veg_xs, veg_means_smooth - veg_stds_smooth, veg_means_smooth + veg_stds_smooth, 
                            facecolor='green', alpha=0.1, label='Std Dev')

        # plot vegetation legend
        plt.legend(loc='upper left')

        # plot precipitation if availible
        if 'total_precipitation' in df.columns:
            # handle the case where vegetation and precipitation have mismatched NaNs
            precip_df = df.dropna(subset=['total_precipitation'])
            precip_ys = precip_df.total_precipitation

            # get precipitation x values to datetime objects
            try:
                precip_xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in precip_df.date]
            except:
                # if the time series has been resampled the index is a TimeStamp object
                precip_xs =  [datetime.datetime.strptime(d._date_repr,'%Y-%m-%d').date() for d in precip_df.date]

            # duplicate axis for preciptation
            ax2 = ax.twinx()
            color = 'tab:blue'
            ax2.set_ylabel(f'Precipitation', color=color, fontsize=14)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim([min(precip_ys)-1*np.array(precip_ys).std(), max(precip_ys)+2*np.array(precip_ys).std()])

            # plot precipitation
            ax2.plot(precip_xs, precip_ys, linewidth=2, color=color, alpha=0.75)

            # add veg-precip correlation
            raw_corr = veg_means.corr(precip_ys)
            smoothed_corr = veg_means_smooth.corr(precip_ys)
            textstr = f'$r={smoothed_corr:.2f}$ (${raw_corr:.2f}$ unsmoothed)'
            ax2.text(0.13, 0.95, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top')

        # plot second vegetation time series if availible
        if veg_prefix_b:
            # function to help plot many y axes
            def make_patch_spines_invisible(ax):
                ax.set_frame_on(True)
                ax.patch.set_visible(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)

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

        # add autoregression info
        unsmoothed_ar1, unsmoothed_ar1_se = get_AR1_parameter_estimate(veg_means)
        smoothed_ar1, smoothed_ar1_se = get_AR1_parameter_estimate(veg_means_smooth)
        textstr = f'AR$(1)={smoothed_ar1:.2f} \pm {smoothed_ar1_se:.2f}$ (${unsmoothed_ar1:.2f} \pm {unsmoothed_ar1_se:.2f}$ unsmoothed)'
        ax.text(0.45, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')

        # add Kendall tau
        tau, p = get_kendell_tau(veg_means)
        tau_smooth, p_smooth = get_kendell_tau(veg_means_smooth)
        kendall_tau_dict = {}
        kendall_tau_dict['Kendall_tau'] = {'unsmoothed': {'tau': tau, 'p': p}, 'smoothed': {'tau': tau_smooth, 'p': p_smooth}}
        write_to_json(os.path.join(output_dir, veg_prefix+'_kendall_tau.json'), kendall_tau_dict)
        textstr = f'$\\tau,~p$-$\\mathrm{{value}}={tau_smooth:.2f}$, ${p:.2f}$ (${tau:.2f}$, ${p_smooth:.2f}$ unsmoothed)'
        ax.text(0.13, 0.85, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')

        # layout
        sns.set_style("white")
        fig.tight_layout()
        
        # save the plot
        output_filename = veg_prefix + '-time-series' + filename_suffix + '.png'
        plt.savefig(os.path.join(output_dir, output_filename), dpi=plot_dpi)


    # make plots for selected columns
    for column in df.columns:
        if 'offset50_mean' in column:
            veg_prefix = column.split('_')[0]
            print(f'\nPlotting {veg_prefix} time series.')
            make_plot(df, veg_prefix, output_dir)

    #print(df.columns.str.contains('offset50_mean'))
    #print(np.sum(df.columns.str.contains('offset50_mean')))


def plot_autocorrelation_function(df, output_dir, filename_suffix=''):
    """
    Given a time series DataFrames (constructed with `make_time_series`),
    plot the autocorrelation function relevant columns.

    Parameters
    ----------
    df : DataFrame
        Time series DataFrame.

    output_dir : str
        Directory to save the plots in.
    """

    def make_plots(series, output_dir, filename_suffix=''):

        # make the full autocorrelation function plot
        plt.figure(figsize=(8,5))
        pd.plotting.autocorrelation_plot(series, label=series.name)
        plt.legend()

        # save the plot
        output_filename = series.name + '-autocorrelation-function' + filename_suffix + '.png'
        plt.savefig(os.path.join(output_dir, output_filename), dpi=plot_dpi)

        # use statsmodels for partial autocorrelation
        from statsmodels.graphics.tsaplots import plot_pacf
        _, ax = plt.subplots(figsize=(8,5))
        plot_pacf(series, label=series.name, ax=ax, zero=False)
        plt.ylim([-1.0, 1.0])
        plt.xlabel('Lag')
        plt.ylabel('Partial Autocorrelation')

        # save the plot
        output_filename = series.name + '-partial-autocorrelation-function' + filename_suffix + '.png'
        plt.savefig(os.path.join(output_dir, output_filename), dpi=plot_dpi)
        
    # make plots for selected columns
    for column in df.columns:
        if 'offset50' in column and 'mean' in column or 'total_precipitation' in column:
            print(f'Plotting autocorrelation functions for "{column}"...')
            make_plots(df[column].dropna(), output_dir, filename_suffix=filename_suffix)


def plot_cross_correlations(df, output_dir):
    """
    Plot a scatterplot matrix showing correlations between vegetation
    and precipitation time series, with different lags. Additionally
    write out the correlations as a function of the lag for later use.

    Parameters
    ----------
    df: DataFrame
        Time-series data.
    output_dir : str
        Directory to save the plot in.
    """

    # check precipitation time series present
    if 'total_precipitation' not in df.columns:
        print('Missing precipitation time series, skipping cross correlation plots.')
        return


    def make_plot(veg_ys, precip_ys, output_dir):

        # set up
        lags = 9
        correlations = []

        # make a new df to ensure NaN veg values are explicit
        df_ = pd.DataFrame()
        df_['precip'] = precip_ys
        df_['offset50'] = veg_ys

        # create fig
        _, axs = plt.subplots(3, 3, sharex='col', sharey='row', 
                                figsize=(8, 8))

        # loop through offsets
        for lag in range(0, lags):

            # select the relevant Axis object
            ax = axs.flat[lag]

            # format this subplot
            ax.set_title(f'$t-{lag}$')
            ax.grid(False)

            # plot data
            lagged_data = df_['offset50'].shift(-lag)
            corr = precip_ys.corr(lagged_data)
            correlations.append(round(corr,4))
            sns.regplot(precip_ys, lagged_data, label=f'$r={corr:.2f}$', ax=ax)
            
            # format axis label
            if lag < 6:
                ax.set_xlabel('')
            if lag % 3 != 0:
                ax.set_ylabel('')
                
            ax.legend()

        plt.tight_layout()

        # save the plot
        output_filename = veg_ys.name + '-scatterplot-matrix.png'
        plt.savefig(os.path.join(output_dir, output_filename), dpi=plot_dpi)

        # write out correlations as a function of lag
        correlations_dict = {veg_ys.name + '_lagged_correlation': correlations}
        write_to_json(os.path.join(output_dir, 'lagged_correlations.json'), correlations_dict)
    
    # make plots for selected columns
    for column in df.columns:
        if 'offset50' in column and 'mean' in column:
            print(f'Plotting cross correlation matrix for "{column}"...')
            make_plot(df[column], df['total_precipitation'], output_dir)


def plot_feature_vector(dfs, output_dir):
    """
    Plot the feature vectors from the network centrality
    output of any vegetation DataFrames in `df`.

    Parameters
    ----------
    dfs : dict of DataFrame
        Time-series data.

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
            plt.savefig(os.path.join(output_dir, output_filename), dpi=plot_dpi)


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
            print(f'Plotting feature vector "{os.path.abspath(output_filename)}"...')
            plt.savefig(os.path.join(output_dir, output_filename), dpi=plot_dpi)
            #plt.show()
          

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

def do_stl_decomposition(dfs, period, output_dir):
    """
     Run the STL decomposition and plot the results network centrality and
     precipitation DataFrames in `df`.

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
