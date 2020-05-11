"""
Plotting code.
"""

import datetime
import os
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

from pyveg.src.data_analysis_utils import (
    get_AR1_parameter_estimate, 
    get_kendell_tau, 
    write_to_json, 
    stl_decomposition,
    get_max_lagged_cor,
    get_datetime_xs
)

# globally set image quality
DPI = 150


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

    def make_plot(df, veg_prefix, output_dir, veg_prefix_b=None, smoothing_option='smooth'):

        # handle the case where vegetation and precipitation have mismatched NaNs
        veg_df = df.dropna(subset=[veg_prefix+'_offset50_mean'])

        # get vegetation x values to datetime objects
        veg_xs = get_datetime_xs(veg_df)

        # get vegetation y values
        veg_means = veg_df[veg_prefix + '_offset50_mean']
        veg_std = veg_df[veg_prefix + '_offset50_std']

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
        if any([smoothing_option in c and veg_prefix in c for c in veg_df.columns]):

            # get smoothed mean, std
            veg_means_smooth = veg_df[veg_prefix+'_offset50_'+smoothing_option+'_mean']
            veg_stds_smooth = veg_df[veg_prefix+'_offset50_'+smoothing_option+'_std']

            # plot smoothed vegetation means and std
            ax.plot(veg_xs, veg_means_smooth, marker='o', markersize=7, 
                    markeredgecolor=(0.9172, 0.9627, 0.9172), markeredgewidth=2,
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
            precip_xs = get_datetime_xs(precip_df)

            # duplicate axis for preciptation
            ax2 = ax.twinx()
            color = 'tab:blue'
            ax2.set_ylabel(f'Precipitation [mm]', color=color, fontsize=14)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim([min(precip_ys)-1*np.array(precip_ys).std(), max(precip_ys)+2*np.array(precip_ys).std()])

            # plot precipitation
            ax2.plot(precip_xs, precip_ys, linewidth=2, color=color, alpha=0.75)

            # add veg-precip correlation
            max_corr_smooth, max_corr = get_max_lagged_cor(os.path.dirname(output_dir), veg_prefix)
            textstr = f'$r_{{t-{max_corr_smooth[1]}}}={max_corr_smooth[0]:.2f}$ '
            textstr += f'($r_{{t-{max_corr[1]}}}={max_corr[0]:.2f}$ unsmoothed)'
            
            # old correlation just calculates the 0-lag correlation
            #raw_corr = veg_means.corr(precip_ys)
            #smoothed_corr = veg_means_smooth.corr(precip_ys)
            #textstr = f'$r={smoothed_corr:.2f}$ (${raw_corr:.2f}$ unsmoothed)'
            ax2.text(0.13, 0.95, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top')

        # plot second vegetation time series if availible
        if veg_prefix_b:

            # handle the case where vegetation and precipitation have mismatched NaNs
            veg_df_b = df.dropna(subset=[veg_prefix_b+'_offset50_mean'])

            # get vegetation x values to datetime objects
            veg_xs_b = get_datetime_xs(veg_df_b)

            # get vegetation y values
            veg_means_b = veg_df_b[veg_prefix_b+'_offset50_mean']
            #veg_std_b = veg_df[veg_prefix_b+'_offset50_std']
            veg_means_smooth_b = veg_df_b[veg_prefix_b+'_offset50_smooth_mean']
            veg_stds_smooth_b = veg_df_b[veg_prefix_b+'_offset50_smooth_std']

            # plot secondary time series
            ax3 = ax.twinx()
            ax3.spines["left"].set_position(("axes", -0.08))
            ax3.spines["left"].set_visible(True)
            color = 'tab:purple'
            ax3.set_ylabel(veg_prefix_b + ' Offset50', color=color, fontsize=14)
            ax3.tick_params(axis='y', labelcolor=color)
            ax3.yaxis.tick_left()
            ax3.yaxis.set_label_position('left')
            ax3.set_ylim([veg_means.min() - 1*veg_std.max(), veg_means.max() + 3*veg_std.max()])

            # plot unsmoothed vegetation means
            ax.plot(veg_xs_b, veg_means_b, label='Unsmoothed', linewidth=1, color='indigo', linestyle='dashed', alpha=0.2)

            # plot smoothed vegetation means and std
            ax3.plot(veg_xs_b, veg_means_smooth_b, marker='o', markersize=7, 
                    markeredgecolor=(0.8172, 0.7627, 0.9172), markeredgewidth=2, 
                    label='Smoothed', linewidth=2, color=color)

            ax3.fill_between(veg_xs_b, veg_means_smooth_b - veg_stds_smooth_b, veg_means_smooth_b + veg_stds_smooth_b, 
                            facecolor='tab:purple', alpha=0.1, label='Std Dev')

            # add veg-veg correlation
            vegveg_corr = veg_means.corr(veg_means_b)
            vegveg_corr_smooth = veg_means_smooth.corr(veg_means_smooth_b)
            textstr = f'$r_{{vv}}={vegveg_corr_smooth:.2f}$ (${vegveg_corr:.2f}$ unsmoothed)'
            ax2.text(0.55, 0.85, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top')

            # update prefix for filename use
            veg_prefix = veg_prefix + '+' + veg_prefix_b

        # add autoregression info
        veg_means.index = veg_df.date
        unsmoothed_ar1, unsmoothed_ar1_se = get_AR1_parameter_estimate(veg_means)
        if any(['smooth' in c and veg_prefix in c for c in veg_df.columns]):
            veg_means_smooth.index = veg_df.date
            smoothed_ar1, smoothed_ar1_se = get_AR1_parameter_estimate(veg_means_smooth)
        else:
            smoothed_ar1, smoothed_ar1_se = np.NaN, np.NaN
        ar1_dict = {}
        ar1_dict['AR1'] = {'unsmoothed': {'param': unsmoothed_ar1, 'se': unsmoothed_ar1_se}, 'smoothed': {'param': smoothed_ar1, 'se': smoothed_ar1_se}}
        write_to_json(os.path.join(output_dir, veg_prefix+'_stats.json'), ar1_dict)
        textstr = f'AR$(1)={smoothed_ar1:.2f} \pm {smoothed_ar1_se:.2f}$ (${unsmoothed_ar1:.2f} \pm {unsmoothed_ar1_se:.2f}$ unsmoothed)'
        ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')

        # add Kendall tau
        tau, p = get_kendell_tau(veg_means)
        if any(['smooth' in c and veg_prefix in c for c in veg_df.columns]):
            tau_smooth, p_smooth = get_kendell_tau(veg_means_smooth)
        else:
            tau_smooth, p_smooth = np.NaN, np.NaN

        kendall_tau_dict = {}
        kendall_tau_dict['Kendall_tau'] = {'unsmoothed': {'tau': tau, 'p': p}, 'smoothed': {'tau': tau_smooth, 'p': p_smooth}}
        write_to_json(os.path.join(output_dir, veg_prefix+'_stats.json'), kendall_tau_dict)
        textstr = f'$\\tau,~p$-$\\mathrm{{value}}={tau_smooth:.2f}$, ${p:.2f}$ (${tau:.2f}$, ${p_smooth:.2f}$ unsmoothed)'
        ax.text(0.13, 0.85, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')

        # layout
        sns.set_style('white')
        fig.tight_layout()

        filename_suffix = '_' + smoothing_option

        # save the plot
        output_filename = veg_prefix + '-time-series' + filename_suffix + '.png'
        plt.savefig(os.path.join(output_dir, output_filename), dpi=DPI)
        plt.close(fig)

    # make output dir if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # make plots for selected columns
    for column in df.columns:
        if 'offset50_mean' in column:
            veg_prefix = column.split('_')[0]
            print(f'Plotting {veg_prefix} time series.')
            make_plot(df, veg_prefix, output_dir)
            make_plot(df, veg_prefix, output_dir, smoothing_option='smooth_res')

    # if we have two vegetation time series availible, plot them both
    if np.sum(df.columns.str.contains('offset50_mean')) == 2:
        veg_columns = df.columns[np.where(df.columns.str.contains('offset50_mean'))].values
        veg_prefixes = [c.split('_')[0] for c in veg_columns]
        assert( len(veg_prefixes) == 2 )
        make_plot(df, veg_prefixes[0], output_dir, veg_prefix_b=veg_prefixes[1])


def plot_ndvi_time_series(df, output_dir):
    def make_plot(df, veg_prefix, output_dir):
        veg_df = df.dropna(subset=[veg_prefix+'_veg_ndvi_mean_mean'])

        # get vegetation x values to datetime objects
        veg_xs = get_datetime_xs(veg_df)

        # get vegetation y values
        veg_means = veg_df[veg_prefix + '_veg_ndvi_mean_mean']
        veg_std = veg_df[veg_prefix + '_veg_ndvi_mean_std']
        veg_std = veg_df[veg_prefix + '_veg_ndvi_std_mean']

        # create a figure
        fig, ax = plt.subplots(figsize=(15, 4.5))
        plt.xlabel('Time', fontsize=14)

        # set up veg y axis
        color = 'tab:green'
        ax.set_ylabel(f'{veg_prefix} NDVI', color=color, fontsize=14)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim([veg_means.min() - 1*veg_std.max(), veg_means.max() + 3*veg_std.max()])

        # plot ndvi
        ax.plot(veg_xs, veg_means, marker='o', markersize=7, 
                markeredgecolor=(0.9172, 0.9627, 0.9172), markeredgewidth=2,
                label='Smoothed', linewidth=2, color='green')

        ax.fill_between(veg_xs, veg_means - veg_std, veg_means + veg_std, 
                        facecolor='green', alpha=0.1, label='Std Dev')
                        
        # plot precipitation if availible
        if 'total_precipitation' in df.columns:
            # handle the case where vegetation and precipitation have mismatched NaNs
            precip_df = df.dropna(subset=['total_precipitation'])
            precip_ys = precip_df.total_precipitation

            # get precipitation x values to datetime objects
            precip_xs = get_datetime_xs(precip_df)

            # duplicate axis for preciptation
            ax2 = ax.twinx()
            color = 'tab:blue'
            ax2.set_ylabel(f'Precipitation [mm]', color=color, fontsize=14)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim([min(precip_ys)-1*np.array(precip_ys).std(), max(precip_ys)+2*np.array(precip_ys).std()])

            # plot precipitation
            ax2.plot(precip_xs, precip_ys, linewidth=2, color=color, alpha=0.75)


        # layout
        sns.set_style('white')
        fig.tight_layout()

        # save the plot
        output_filename = veg_prefix + '-ndvi-time-series.png'
        plt.savefig(os.path.join(output_dir, output_filename), dpi=DPI)
        plt.close(fig)

    # make output dir if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # make plots for selected columns
    for column in df.columns:
        if 'veg_ndvi_mean' in column:
            veg_prefix = column.split('_')[0]
            print(f'Plotting {veg_prefix} time series.')
            make_plot(df, veg_prefix, output_dir)

    
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
        plt.savefig(os.path.join(output_dir, output_filename), dpi=DPI)

        # use statsmodels for partial autocorrelation
        from statsmodels.graphics.tsaplots import plot_pacf
        fig, ax = plt.subplots(figsize=(8,5))
        plot_pacf(series, label=series.name, ax=ax, zero=False)
        plt.ylim([-1.0, 1.0])
        plt.xlabel('Lag')
        plt.ylabel('Partial Autocorrelation')

        # save the plot
        output_filename = series.name + '-partial-autocorrelation-function' + filename_suffix + '.png'
        plt.savefig(os.path.join(output_dir, output_filename), dpi=DPI)
        plt.close(fig)
        
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
        fig, axs = plt.subplots(3, 3, sharex='col', sharey='row', 
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
        plt.savefig(os.path.join(output_dir, output_filename), dpi=DPI)
        plt.close(fig)

        # write out correlations as a function of lag
        correlations_dict = {veg_ys.name + '_lagged_correlation': correlations}
        write_to_json(os.path.join(output_dir, 'lagged_correlations.json'), correlations_dict)
    
    # make plots for selected columns
    for column in df.columns:
        if 'offset50' in column and 'mean' in column:
            print(f'Plotting cross correlation matrix for "{column}"...')
            make_plot(df[column], df['total_precipitation'], output_dir)


def plot_feature_vector(output_dir):
    """
    Read feature vectors from csv (if they exist) and then
    make feature vector plots.

    Parameters
    ----------
    output_dir : str
        Directory to save the plot in.
    """

    # assume feature vectors have been saved in the above location
    fv_dir = os.path.join(os.path.dirname(output_dir), 'processed_data')
    if not os.path.exists(fv_dir):
        print('No feature vectors found, skipping plot!')
        return

    # get feature vectors for different collections
    fvs = [f for f in os.listdir(fv_dir) if '_feature_vectors.csv' in f]
    if len(fvs) == 0:
        print('No feature vectors found, skipping plot!')
        return

    # for each collection
    for fv_filename in fvs:

        # read feature vectors
        df = pd.read_csv(os.path.join(fv_dir, fv_filename)).dropna()

        # percentile columns
        cols = [c for c in df.columns if 'percentile' in c]
        
        # compute feature vector averaged over all sub-images
        feature_vector = df[cols].mean()
        feature_vector_std = df[cols].std()

        # generate x-values
        xs = np.linspace(0,100,len(feature_vector))
        
        # make the plot
        fig, _ = plt.subplots(figsize=(6,5))

        plt.errorbar(xs, feature_vector, marker='o', markersize=5, linestyle='', 
                        yerr=feature_vector_std, color='black', capsize=2, elinewidth=1)

        plt.xlabel('Pixel Rank (%)', fontsize=14)
        plt.ylabel('$X(V-E)$', fontsize=14)
        plt.tight_layout()

        # save the plot
        output_filename = fv_filename.split('_')[0] + '-feature-vector-summary.png'
        print(f'Plotting feature vector "{os.path.abspath(output_filename)}"...')
        plt.savefig(os.path.join(output_dir, output_filename), dpi=DPI)
        plt.close(fig)

        feature_vecs = []
        feature_vecs_stds = []
        offset50s = []
        dates = []

        # loop through time points
        for date, group in df.groupby('date'):
            
            # calculate feature vector and offset50
            feature_vector = group.mean()[cols]
            feature_vecs.append(feature_vector)
            feature_vecs_stds.append(group.std()[cols])
            offset50s.append((feature_vector[-1] - feature_vector[len(feature_vector)//2]))
            dates.append(date)

        # get max and min
        imax = np.argmax(np.array(offset50s))
        imin = np.argmin(np.array(offset50s))

        max_fv = feature_vecs[imax]
        max_fv_std = feature_vecs_stds[imax]
        max_date = dates[imax]
        min_fv = feature_vecs[imin]
        min_fv_std = feature_vecs_stds[imin]
        min_date = dates[imin]

        # plot the min/max veg feature vectors
        fig, _ = plt.subplots(figsize=(6,5))

        # add to plot
        plt.errorbar(xs, max_fv, marker='o', markersize=5, linestyle='', label=f'max veg: {max_date}',
                        yerr=max_fv_std, color='tab:green', capsize=2, elinewidth=1)
        plt.errorbar(xs, min_fv, marker='o', markersize=5, linestyle='', label=f'min veg: {min_date}',
                        yerr=min_fv_std, color='tab:red', capsize=2, elinewidth=1)

        # format plot
        plt.xlabel('Pixel Rank (%)', fontsize=14)
        plt.ylabel('$X(V-E)$', fontsize=14)
        plt.legend()
        plt.tight_layout()

        # save the plot
        output_filename = fv_filename.split('_')[0] + '-feature-vector-minmax.png'
        print(f'Plotting minmax feature vector "{os.path.abspath(output_filename)}"...')
        plt.savefig(os.path.join(output_dir, output_filename), dpi=DPI)
        plt.close(fig)


def plot_stl_decomposition(df, period, output_dir):
    """
     Run the STL decomposition and plot the results network centrality and
     precipitation DataFrames in `df`.

     Parameters
     ----------
     df : DataFrame
         The time-series results.
     period : float
        Periodicity to model.
     output_dir : str
         Directory to save the plot in.
     """

    def make_plot(df, column, output_dir):
        """
        Plot STL decomposition results.

        Parameters
        ----------
        df : DataFrame
            The input time-series.
        column : str
            Column name to run STL on.
        output_dir : str
            Directory to save the plot in.
        """

        # run fit
        res = stl_decomposition(df[column], period)

        # concert x values to datetime objects
        xs = get_datetime_xs(df)

        # formatting
        default_figsize = plt.rcParams['figure.figsize']
        default_fontsize = plt.rcParams['font.size']
        plt.rc('figure', figsize=(20, 8))
        plt.rc('font', size=15)

        fig = res.plot()
        ax_list = fig.axes
        for ax in ax_list[:-1]:
            ax.tick_params(labelbottom=False)

        # set xlabel with datetime object
        #ax_list[-1].set_xticklabels(xs, rotation=0, va="center")

        # save plot
        filename = os.path.join(output_dir, column+'_STL_decomposition.png')
        plt.savefig(filename, dpi=DPI)
        plt.close(fig)

        # undo rc changes
        plt.rc('figure', figsize=default_figsize)
        plt.rc('font', size=default_fontsize)
    
    # make output dir if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # make plots for selected columns
    for column in df.columns:
        if 'offset50' in column and 'mean' in column or 'total_precipitation' in column:
            
            print(f'Plotting STL decomposition for "{column}"...')
            
            # produce plot
            make_plot(df.dropna(), column, output_dir)


def plot_moving_window_analysis(df, output_dir, filename_suffix=""):
    """
    Given a moving window time series DataFrame, plot the time series 
    of AR1 and Variance.

    Parameters
    ----------
    df : DataFrame
        The time-series results for variance and AR1.
    output_dir : str
        Directory to save the plot in.
    filename_suffix: str
        Add suffix string to file name
    """

    def make_plot(df, column, output_dir, smoothing_option):
        """
        Parameters
        ----------
        df : DataFrame
            The time-series results for variance and AR1.
        column : str
            Column name an offset50 variance column in df.
        output_dir : str
            Directory to save the plot in.
        smoothing_option: str
            Label for smoothing variable to be used
        """

        # get short string prefix on column name
        collection_prefix = column.split('_')[0] if 'offset50' in column else 'precipitation'
        
        # hand mismatched NaNs
        ar1_df = df.dropna(subset=[column.replace('var', 'ar1')])
        var_df = df.dropna(subset=[column])

        # extract x values and convert to datetime objects
        ar1_xs = get_datetime_xs(ar1_df)
        var_xs = get_datetime_xs(var_df)

        # extract individual time series
        variance = var_df[column]
        ar1 = ar1_df[column.replace('var', 'ar1')]
        ar1_se = ar1_df[column.replace('var', 'ar1_se')]

        if any([smoothing_option in c for c in df.columns]):
            variance_smooth = var_df[column.replace('offset50_mean', 'offset50_' + smoothing_option + '_mean')]
            ar1_smooth = ar1_df[column.replace('var', 'ar1').replace('offset50_mean', 'offset50_' + smoothing_option + '_mean')]
            ar1_se_smooth = ar1_df[column.replace('var', 'ar1_se').replace('offset50_mean', 'offset50_' + smoothing_option + '_mean')]

        # create a figure
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.xlabel('Time', fontsize=12)

        # set up veg y axis
        color = 'tab:blue'
        ax.set_ylabel(f'{collection_prefix} AR1', color=color, fontsize=12)
        ax.tick_params(axis='y', labelcolor=color)

        # plot unsmoothed vegetation ar1 and std
        ax.plot(ar1_xs, ar1, label='AR1', linewidth=2, color='tab:blue')
        ax.fill_between(ar1_xs, ar1 - ar1_se, ar1 + ar1_se,
                        facecolor='blue', alpha=0.1, label='AR1 SE')

        if any([smoothing_option in c for c in df.columns]):
            # plot smoothed vegetation ar1 and std
            ax.plot(ar1_xs, ar1_smooth, label='AR1 Smoothed', linewidth=2, color='tab:blue', linestyle='dotted')
            ax.fill_between(ar1_xs, ar1_smooth - ar1_se_smooth, ar1_smooth + ar1_se_smooth,
                            facecolor='none', alpha=0.15, label='AR1 SE Smoothed', hatch='X', edgecolor='tab:blue')

        # set y lim
        ax.set_ylim([min(ar1-ar1_se)-0.8*max(ar1+ar1_se), 1.8*max(ar1+ar1_se)])

        # plot legend
        plt.legend(loc='upper left')

        # duplicate x-axis for variance
        ax2 = ax.twinx()
        color = 'tab:red'
        ax2.set_ylabel(f'{collection_prefix} Variance', color=color, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color)

        # plot variance
        ax2.plot(var_xs, variance, linewidth=2, color=color, alpha=0.75, label='Variance')
        if any([smoothing_option in c for c in df.columns]):
            ax2.plot(var_xs, variance_smooth, linewidth=2, color=color, alpha=0.75, 
                     linestyle='dotted', label='Variance Smoothed')

        # set y lim
        ax2.set_ylim([0, 2*max(variance)])


        # add legend
        plt.legend(loc='lower left')

        # add Kendall tau

        tau, p = get_kendell_tau(ar1)
        tau_var, p_var = get_kendell_tau(variance)

        tau_smooth, p_smooth = get_kendell_tau(ar1_smooth)
        tau_var_smooth, p_var_smooth = get_kendell_tau(variance_smooth)

        # add to plot
        textstr = f'AR1 Kendall $\\tau,~p$-$\\mathrm{{value}}={tau_smooth:.2f}$, ${p_smooth:.2f}$'
        textstr += f' (${tau:.2f}$, ${p:.2f}$ unsmoothed)'

        ax2.text(0.43, 0.95, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top')
        textstr = f'Variance Kendall $\\tau,~p$-$\\mathrm{{value}}={tau_var_smooth:.2f}$, ${p_var_smooth:.2f}$'
        textstr += f' (${tau_var:.2f}$, ${p_var:.2f}$ unsmoothed)'

        ax2.text(0.43, 0.85, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top')
        # layout
        fig.tight_layout()

        # save the plot
        output_filename = collection_prefix + '-moving-window-AR1-var' + smoothing_option + '.png'
        print(f'Plotting {collection_prefix} moving window time series...')
        plt.savefig(os.path.join(output_dir, output_filename), dpi=DPI)
        plt.close(fig)


    for column in df.columns:
        if (('offset50_mean' in column or 'total_precipitation' in column) and 
             'var' in column):
            make_plot(df, column, output_dir, 'smooth_res')
